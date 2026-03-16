const std = @import("std");
const testing = std.testing;

/// SwissTable: Group-based open addressing hash table with control bytes.
/// Inspired by Google's Abseil SwissTable design.
///
/// Features:
/// - Group-based probing (8 slots per group)
/// - Control bytes for SIMD-friendly lookup (emulated without SIMD intrinsics)
/// - Quadratic probing across groups
/// - Tombstone-free deletion via backward shift
///
/// Time Complexity:
/// - insert: O(1) expected, O(n) worst case (rehash)
/// - get: O(1) expected
/// - remove: O(1) expected
/// - Space: O(n) with ~87.5% max load factor
pub fn SwissTable(
    comptime K: type,
    comptime V: type,
    comptime Context: type,
    comptime hash_fn: fn (ctx: Context, key: K) u64,
    comptime eql_fn: fn (ctx: Context, a: K, b: K) bool,
) type {
    return struct {
        const Self = @This();

        /// Number of slots per group (must be power of 2 for efficient modulo)
        const GROUP_SIZE = 8;

        /// Control byte states
        const EMPTY: u8 = 0b1111_1111; // -1 in signed i8
        const DELETED: u8 = 0b1111_1110; // -2 in signed i8 (tombstone, but we use backward shift instead)

        /// Maximum load factor before resize (87.5% = 7/8)
        const MAX_LOAD_FACTOR_NUMERATOR = 7;
        const MAX_LOAD_FACTOR_DENOMINATOR = 8;

        pub const Entry = struct {
            key: K,
            value: V,
        };

        pub const Iterator = struct {
            table: *const Self,
            index: usize,

            /// Time: O(1) amortized over full iteration
            pub fn next(self: *Iterator) ?Entry {
                while (self.index < self.table.capacity) : (self.index += 1) {
                    if (self.table.ctrl[self.index] < 0x80) { // occupied slot
                        const entry = Entry{
                            .key = self.table.keys[self.index],
                            .value = self.table.values[self.index],
                        };
                        self.index += 1;
                        return entry;
                    }
                }
                return null;
            }
        };

        allocator: std.mem.Allocator,
        ctx: Context,
        keys: []K,
        values: []V,
        ctrl: []u8, // Control bytes: EMPTY, DELETED, or H2 hash (0-127)
        capacity: usize,
        len: usize,

        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: std.mem.Allocator, ctx: Context) !Self {
            return initCapacity(allocator, ctx, GROUP_SIZE);
        }

        /// Time: O(n) | Space: O(n)
        pub fn initCapacity(allocator: std.mem.Allocator, ctx: Context, cap: usize) !Self {
            if (cap == 0) {
                return Self{
                    .allocator = allocator,
                    .ctx = ctx,
                    .keys = &[_]K{},
                    .values = &[_]V{},
                    .ctrl = &[_]u8{},
                    .capacity = 0,
                    .len = 0,
                };
            }

            // Round up to multiple of GROUP_SIZE
            const actual_cap = ((cap + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;

            const keys = try allocator.alloc(K, actual_cap);
            errdefer allocator.free(keys);

            const values = try allocator.alloc(V, actual_cap);
            errdefer allocator.free(values);

            const ctrl = try allocator.alloc(u8, actual_cap);
            @memset(ctrl, EMPTY);

            return Self{
                .allocator = allocator,
                .ctx = ctx,
                .keys = keys,
                .values = values,
                .ctrl = ctrl,
                .capacity = actual_cap,
                .len = 0,
            };
        }

        /// Time: O(n) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.capacity > 0) {
                self.allocator.free(self.keys);
                self.allocator.free(self.values);
                self.allocator.free(self.ctrl);
            }
            self.* = undefined;
        }

        /// Time: O(n) | Space: O(n)
        pub fn clone(self: *const Self) !Self {
            var new = try initCapacity(self.allocator, self.ctx, self.capacity);
            errdefer new.deinit();

            if (self.capacity > 0) {
                @memcpy(new.keys, self.keys);
                @memcpy(new.values, self.values);
                @memcpy(new.ctrl, self.ctrl);
                new.len = self.len;
            }

            return new;
        }

        /// Time: O(1) | Space: O(1)
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        /// Extract H2 hash (7 bits for control byte)
        fn h2(hash: u64) u8 {
            return @truncate(hash & 0x7F); // Use lower 7 bits, keep MSB=0 for occupied
        }

        /// Probe sequence: quadratic probing across groups
        fn probeStart(hash: u64, mask: usize) usize {
            return @as(usize, @truncate(hash)) & mask;
        }

        fn probeNext(index: usize, stride: usize, mask: usize) usize {
            return (index + stride) & mask;
        }

        /// Find slot for key (either existing or empty)
        fn findSlot(self: *const Self, key: K, hash: u64) ?usize {
            if (self.capacity == 0) return null;

            const mask = self.capacity - 1;
            const h2_val = h2(hash);
            var idx = probeStart(hash, mask);
            var stride: usize = 1;

            while (stride <= self.capacity) : (stride += GROUP_SIZE) {
                // Check group of GROUP_SIZE slots
                var offset: usize = 0;
                while (offset < GROUP_SIZE) : (offset += 1) {
                    const pos = (idx + offset) & mask;
                    const ctrl_val = self.ctrl[pos];

                    if (ctrl_val == EMPTY) {
                        return null; // Key not found
                    } else if (ctrl_val == h2_val) {
                        // Potential match, verify key
                        if (eql_fn(self.ctx, self.keys[pos], key)) {
                            return pos;
                        }
                    }
                }
                idx = probeNext(idx, stride, mask);
            }
            return null;
        }

        /// Find empty slot for insertion
        fn findEmptySlot(self: *const Self, hash: u64) ?usize {
            if (self.capacity == 0) return null;

            const mask = self.capacity - 1;
            var idx = probeStart(hash, mask);
            var stride: usize = 1;

            while (stride <= self.capacity) : (stride += GROUP_SIZE) {
                var offset: usize = 0;
                while (offset < GROUP_SIZE) : (offset += 1) {
                    const pos = (idx + offset) & mask;
                    if (self.ctrl[pos] == EMPTY) {
                        return pos;
                    }
                }
                idx = probeNext(idx, stride, mask);
            }
            return null;
        }

        /// Time: O(1) expected, O(n) worst case (rehash) | Space: O(1) amortized
        pub fn insert(self: *Self, key: K, value: V) !?V {
            // Check load factor and grow if needed
            const threshold = (self.capacity * MAX_LOAD_FACTOR_NUMERATOR) / MAX_LOAD_FACTOR_DENOMINATOR;
            if (self.len >= threshold) {
                const new_cap = if (self.capacity == 0) GROUP_SIZE else self.capacity * 2;
                try self.grow(new_cap);
            }

            const hash = hash_fn(self.ctx, key);

            // Check if key exists
            if (self.findSlot(key, hash)) |pos| {
                const old_value = self.values[pos];
                self.values[pos] = value;
                return old_value;
            }

            // Find empty slot
            const pos = self.findEmptySlot(hash) orelse return error.TableFull;

            self.keys[pos] = key;
            self.values[pos] = value;
            self.ctrl[pos] = h2(hash);
            self.len += 1;

            return null;
        }

        /// Time: O(1) expected | Space: O(1)
        pub fn get(self: *const Self, key: K) ?V {
            const hash = hash_fn(self.ctx, key);
            if (self.findSlot(key, hash)) |pos| {
                return self.values[pos];
            }
            return null;
        }

        /// Time: O(1) expected | Space: O(1)
        pub fn getPtr(self: *Self, key: K) ?*V {
            const hash = hash_fn(self.ctx, key);
            if (self.findSlot(key, hash)) |pos| {
                return &self.values[pos];
            }
            return null;
        }

        /// Time: O(1) expected | Space: O(1)
        pub fn contains(self: *const Self, key: K) bool {
            const hash = hash_fn(self.ctx, key);
            return self.findSlot(key, hash) != null;
        }

        /// Time: O(1) expected | Space: O(1)
        /// Uses backward shift deletion (tombstone-free)
        pub fn remove(self: *Self, key: K) ?Entry {
            const hash = hash_fn(self.ctx, key);
            const pos = self.findSlot(key, hash) orelse return null;

            const entry = Entry{
                .key = self.keys[pos],
                .value = self.values[pos],
            };

            // Backward shift deletion
            self.ctrl[pos] = EMPTY;
            self.len -= 1;

            // Shift subsequent entries backward until we hit EMPTY or break the probe chain
            const mask = self.capacity - 1;
            var curr = (pos + 1) & mask;
            var prev = pos;

            while (self.ctrl[curr] != EMPTY) : ({
                prev = curr;
                curr = (curr + 1) & mask;
            }) {
                const curr_hash = hash_fn(self.ctx, self.keys[curr]);
                const ideal = probeStart(curr_hash, mask);

                // Check if shifting this entry backward would maintain probe invariant
                // We can shift if the ideal position is before or at prev
                const can_shift = if (ideal <= prev) true else if (prev < curr) (ideal <= prev or ideal > curr) else (ideal <= prev and ideal > curr);

                if (can_shift) {
                    self.keys[prev] = self.keys[curr];
                    self.values[prev] = self.values[curr];
                    self.ctrl[prev] = self.ctrl[curr];
                    self.ctrl[curr] = EMPTY;
                } else {
                    break;
                }
            }

            return entry;
        }

        /// Time: O(1) | Space: O(1)
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .table = self, .index = 0 };
        }

        /// Time: O(n) | Space: O(n)
        fn grow(self: *Self, new_capacity: usize) (std.mem.Allocator.Error || error{TableFull})!void {
            var new_table = try initCapacity(self.allocator, self.ctx, new_capacity);
            errdefer new_table.deinit();

            // Rehash all entries
            for (0..self.capacity) |i| {
                if (self.ctrl[i] < 0x80) { // occupied
                    _ = try new_table.insert(self.keys[i], self.values[i]);
                }
            }

            self.deinit();
            self.* = new_table;
        }

        /// Time: O(n) | Space: O(1)
        /// Validates internal invariants
        pub fn validate(self: *const Self) !void {
            // Check length consistency
            var actual_len: usize = 0;
            for (self.ctrl) |ctrl_val| {
                if (ctrl_val < 0x80) actual_len += 1;
            }
            if (actual_len != self.len) return error.LengthMismatch;

            // Check capacity is multiple of GROUP_SIZE
            if (self.capacity % GROUP_SIZE != 0) return error.InvalidCapacity;

            // Check all entries are findable
            for (0..self.capacity) |i| {
                if (self.ctrl[i] < 0x80) {
                    const key = self.keys[i];
                    const hash = hash_fn(self.ctx, key);
                    const found_pos = self.findSlot(key, hash) orelse return error.EntryNotFindable;
                    if (found_pos != i) return error.EntryPositionMismatch;
                }
            }

            // Check load factor
            const threshold = (self.capacity * MAX_LOAD_FACTOR_NUMERATOR) / MAX_LOAD_FACTOR_DENOMINATOR;
            if (self.len > threshold and self.capacity > 0) return error.LoadFactorExceeded;
        }
    };
}

// -- Tests --

fn testHash(ctx: void, key: u32) u64 {
    _ = ctx;
    return @as(u64, key) *% 0x9e3779b97f4a7c15; // Fibonacci hashing
}

fn testEql(ctx: void, a: u32, b: u32) bool {
    _ = ctx;
    return a == b;
}

test "SwissTable: init and deinit" {
    var table = try SwissTable(u32, []const u8, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    try testing.expect(table.isEmpty());
    try testing.expectEqual(@as(usize, 0), table.count());
    try table.validate();
}

test "SwissTable: insert and get" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    try testing.expectEqual(@as(?u32, null), try table.insert(1, 100));
    try testing.expectEqual(@as(?u32, null), try table.insert(2, 200));
    try testing.expectEqual(@as(?u32, null), try table.insert(3, 300));

    try testing.expectEqual(@as(?u32, 100), table.get(1));
    try testing.expectEqual(@as(?u32, 200), table.get(2));
    try testing.expectEqual(@as(?u32, 300), table.get(3));
    try testing.expectEqual(@as(?u32, null), table.get(999));

    try testing.expectEqual(@as(usize, 3), table.count());
    try table.validate();
}

test "SwissTable: update existing key" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    try testing.expectEqual(@as(?u32, null), try table.insert(1, 100));
    try testing.expectEqual(@as(?u32, 100), try table.insert(1, 999));
    try testing.expectEqual(@as(?u32, 999), table.get(1));
    try testing.expectEqual(@as(usize, 1), table.count());
    try table.validate();
}

test "SwissTable: remove" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    _ = try table.insert(1, 100);
    _ = try table.insert(2, 200);
    _ = try table.insert(3, 300);

    const removed = table.remove(2).?;
    try testing.expectEqual(@as(u32, 2), removed.key);
    try testing.expectEqual(@as(u32, 200), removed.value);

    try testing.expectEqual(@as(?u32, null), table.get(2));
    try testing.expectEqual(@as(?u32, 100), table.get(1));
    try testing.expectEqual(@as(?u32, 300), table.get(3));
    try testing.expectEqual(@as(usize, 2), table.count());
    try table.validate();
}

test "SwissTable: remove nonexistent" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    try testing.expectEqual(@as(@TypeOf(table.remove(999)), null), table.remove(999));
    try table.validate();
}

test "SwissTable: contains" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    _ = try table.insert(42, 123);
    try testing.expect(table.contains(42));
    try testing.expect(!table.contains(999));
    try table.validate();
}

test "SwissTable: iterator" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    _ = try table.insert(1, 10);
    _ = try table.insert(2, 20);
    _ = try table.insert(3, 30);

    var sum_keys: u32 = 0;
    var sum_values: u32 = 0;
    var iter = table.iterator();
    while (iter.next()) |entry| {
        sum_keys += entry.key;
        sum_values += entry.value;
    }

    try testing.expectEqual(@as(u32, 6), sum_keys);
    try testing.expectEqual(@as(u32, 60), sum_values);
    try table.validate();
}

test "SwissTable: clone" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    _ = try table.insert(1, 100);
    _ = try table.insert(2, 200);

    var cloned = try table.clone();
    defer cloned.deinit();

    try testing.expectEqual(@as(?u32, 100), cloned.get(1));
    try testing.expectEqual(@as(?u32, 200), cloned.get(2));
    try testing.expectEqual(@as(usize, 2), cloned.count());

    // Modify clone, original unchanged
    _ = try cloned.insert(3, 300);
    try testing.expectEqual(@as(?u32, null), table.get(3));
    try testing.expectEqual(@as(?u32, 300), cloned.get(3));

    try table.validate();
    try cloned.validate();
}

test "SwissTable: grow on high load factor" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    const initial_cap = table.capacity;

    // Insert enough to trigger grow (87.5% load factor)
    for (0..100) |i| {
        _ = try table.insert(@intCast(i), @intCast(i * 10));
    }

    try testing.expect(table.capacity > initial_cap);
    try testing.expectEqual(@as(usize, 100), table.count());

    // Verify all entries are still accessible
    for (0..100) |i| {
        const key: u32 = @intCast(i);
        const expected: u32 = @intCast(i * 10);
        try testing.expectEqual(@as(?u32, expected), table.get(key));
    }

    try table.validate();
}

test "SwissTable: stress test with 1000 insertions" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    for (0..1000) |i| {
        const key: u32 = @intCast(i);
        _ = try table.insert(key, key * 2);
    }

    try testing.expectEqual(@as(usize, 1000), table.count());

    for (0..1000) |i| {
        const key: u32 = @intCast(i);
        try testing.expectEqual(@as(?u32, key * 2), table.get(key));
    }

    try table.validate();
}

test "SwissTable: remove with backward shift" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    // Insert multiple entries that may probe
    for (0..50) |i| {
        const key: u32 = @intCast(i);
        _ = try table.insert(key, key);
    }

    // Remove some entries
    for (0..50) |i| {
        if (i % 3 == 0) {
            const key: u32 = @intCast(i);
            _ = table.remove(key);
        }
    }

    // Verify remaining entries are still findable
    for (0..50) |i| {
        const key: u32 = @intCast(i);
        if (i % 3 == 0) {
            try testing.expectEqual(@as(?u32, null), table.get(key));
        } else {
            try testing.expectEqual(@as(?u32, key), table.get(key));
        }
    }

    try table.validate();
}

test "SwissTable: memory leak check" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    for (0..100) |i| {
        _ = try table.insert(@intCast(i), @intCast(i));
    }
    try testing.expectEqual(@as(usize, 100), table.count());

    // Verify all keys are accessible
    for (0..100) |i| {
        try testing.expectEqual(@as(?u32, @intCast(i)), table.get(@intCast(i)));
    }

    // std.testing.allocator will detect leaks automatically
}

test "SwissTable: getPtr modification" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).init(testing.allocator, {});
    defer table.deinit();

    _ = try table.insert(1, 100);
    const ptr = table.getPtr(1).?;
    ptr.* = 999;

    try testing.expectEqual(@as(?u32, 999), table.get(1));
    try table.validate();
}

test "SwissTable: capacity is multiple of GROUP_SIZE" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).initCapacity(testing.allocator, {}, 10);
    defer table.deinit();

    try testing.expect(table.capacity % 8 == 0);
    try testing.expect(table.capacity >= 10);
    try table.validate();
}

test "SwissTable: zero capacity init" {
    var table = try SwissTable(u32, u32, void, testHash, testEql).initCapacity(testing.allocator, {}, 0);
    defer table.deinit();

    try testing.expectEqual(@as(usize, 0), table.capacity);
    try testing.expect(table.isEmpty());

    // Should grow on first insert
    _ = try table.insert(1, 100);
    try testing.expect(table.capacity > 0);
    try testing.expectEqual(@as(?u32, 100), table.get(1));
    try table.validate();
}
