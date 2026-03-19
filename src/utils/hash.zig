/// Hashing Utilities
///
/// Generic hash functions for common types to reduce boilerplate when creating
/// hash-based containers. These follow Zig's std.hash.Wyhash convention.
///
/// Example:
/// ```zig
/// const map = RobinHoodHashMap(Point, i32, void, hash.auto(Point)).init(allocator, {});
/// ```

const std = @import("std");

/// Returns an auto-hash function for any type that implements std.hash.autoHash.
/// Works with integers, floats, structs, arrays, slices, and most Zig types.
///
/// Time: O(size of T) | Space: O(1)
pub fn auto(comptime K: type) fn (ctx: void, key: K) u64 {
    return struct {
        pub fn hash(_: void, key: K) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key);
            return hasher.final();
        }
    }.hash;
}

/// Returns a hash function for string keys.
///
/// Time: O(|key|) | Space: O(1)
pub fn string(_: void, key: []const u8) u64 {
    return std.hash.Wyhash.hash(0, key);
}

/// Returns a hash function that combines hashes of tuple fields.
/// Useful for composite keys.
///
/// Note: For string fields, use the string() function directly or provide a custom hash.
/// Time: O(1) per field | Space: O(1)
pub fn tuple2(comptime T1: type, comptime T2: type) fn (ctx: void, key: struct { T1, T2 }) u64 {
    return struct {
        pub fn hash(_: void, key: struct { T1, T2 }) u64 {
            var hasher = std.hash.Wyhash.init(0);
            // For slice types, hash the content directly
            if (@typeInfo(T1) == .pointer) {
                const ptr_info = @typeInfo(T1).pointer;
                if (ptr_info.size == .Slice) {
                    hasher.update(key[0]);
                } else {
                    std.hash.autoHash(&hasher, key[0]);
                }
            } else {
                std.hash.autoHash(&hasher, key[0]);
            }
            if (@typeInfo(T2) == .pointer) {
                const ptr_info = @typeInfo(T2).pointer;
                if (ptr_info.size == .Slice) {
                    hasher.update(key[1]);
                } else {
                    std.hash.autoHash(&hasher, key[1]);
                }
            } else {
                std.hash.autoHash(&hasher, key[1]);
            }
            return hasher.final();
        }
    }.hash;
}

/// Returns a hash function for pointers (hashes pointed value, not address).
///
/// Time: O(size of T) | Space: O(1)
pub fn deref(comptime T: type) fn (ctx: void, key: *const T) u64 {
    return struct {
        pub fn hash(_: void, key: *const T) u64 {
            var hasher = std.hash.Wyhash.init(0);
            std.hash.autoHash(&hasher, key.*);
            return hasher.final();
        }
    }.hash;
}

/// Returns a case-insensitive string hash function.
///
/// Time: O(|key|) | Space: O(1)
pub fn stringCaseInsensitive(_: void, key: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(0);
    for (key) |c| {
        hasher.update(&[_]u8{std.ascii.toLower(c)});
    }
    return hasher.final();
}

/// Returns an equality function for any type that supports ==.
/// Time: O(size of T) | Space: O(1)
pub fn eqlAuto(comptime K: type) fn (ctx: void, a: K, b: K) bool {
    return struct {
        pub fn eql(_: void, a: K, b: K) bool {
            return std.meta.eql(a, b);
        }
    }.eql;
}

// Tests

const testing = std.testing;

test "auto hash for integers" {
    const hashFn = auto(i64);

    const h1 = hashFn({}, 42);
    const h2 = hashFn({}, 42);
    const h3 = hashFn({}, 43);

    try testing.expectEqual(h1, h2); // same value = same hash
    try testing.expect(h1 != h3); // different value = different hash (high probability)
}

test "auto hash for structs" {
    const Point = struct { x: i32, y: i32 };
    const hashFn = auto(Point);

    const h1 = hashFn({}, Point{ .x = 10, .y = 20 });
    const h2 = hashFn({}, Point{ .x = 10, .y = 20 });
    const h3 = hashFn({}, Point{ .x = 10, .y = 21 });

    try testing.expectEqual(h1, h2);
    try testing.expect(h1 != h3);
}

test "string hash" {
    const h1 = string({}, "hello");
    const h2 = string({}, "hello");
    const h3 = string({}, "world");

    try testing.expectEqual(h1, h2);
    try testing.expect(h1 != h3);
}

test "tuple2 hash" {
    const hashFn = tuple2(i32, i64);
    const Tuple = struct { i32, i64 };

    const h1 = hashFn({}, Tuple{ 1, 100 });
    const h2 = hashFn({}, Tuple{ 1, 100 });
    const h3 = hashFn({}, Tuple{ 1, 200 });
    const h4 = hashFn({}, Tuple{ 2, 100 });

    try testing.expectEqual(h1, h2);
    try testing.expect(h1 != h3);
    try testing.expect(h1 != h4);
}

test "deref hash" {
    const hashFn = deref(i32);
    const a: i32 = 42;
    const b: i32 = 42;
    const c: i32 = 43;

    const h1 = hashFn({}, &a);
    const h2 = hashFn({}, &b);
    const h3 = hashFn({}, &c);

    try testing.expectEqual(h1, h2); // same value
    try testing.expect(h1 != h3); // different value
}

test "stringCaseInsensitive" {
    const h1 = stringCaseInsensitive({}, "Hello");
    const h2 = stringCaseInsensitive({}, "hello");
    const h3 = stringCaseInsensitive({}, "HELLO");
    const h4 = stringCaseInsensitive({}, "World");

    try testing.expectEqual(h1, h2);
    try testing.expectEqual(h1, h3);
    try testing.expect(h1 != h4);
}

test "integration with RobinHoodHashMap" {
    const RobinHoodHashMap = @import("../containers/hashing/robin_hood_hash_map.zig").RobinHoodHashMap;

    const Point = struct { x: i32, y: i32 };
    var map = try RobinHoodHashMap(Point, []const u8, void, auto(Point), eqlAuto(Point)).init(testing.allocator, {});
    defer map.deinit();

    const p1 = Point{ .x = 10, .y = 20 };
    const p2 = Point{ .x = 30, .y = 40 };

    _ = try map.insert(p1, "first");
    _ = try map.insert(p2, "second");

    try testing.expectEqual(@as(?[]const u8, "first"), map.get(p1));
    try testing.expectEqual(@as(?[]const u8, "second"), map.get(p2));
    try testing.expectEqual(@as(usize, 2), map.count());
}
