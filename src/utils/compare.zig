/// Comparison Utilities
///
/// Generic comparison functions for common types to reduce boilerplate when
/// creating containers. These functions work with standard Zig types and follow
/// the std.math.Order convention.
///
/// Example:
/// ```zig
/// const rbt = RedBlackTree(i64, []const u8, void, compare.ascending(i64)).init(allocator, {});
/// ```

const std = @import("std");

/// Returns a comparison function for ascending order of any orderable type.
/// Works with integers, floats, and any type that implements std.math.order.
///
/// Time: O(1) | Space: O(1)
pub fn ascending(comptime T: type) fn (ctx: void, a: T, b: T) std.math.Order {
    return struct {
        pub fn order(_: void, a: T, b: T) std.math.Order {
            return std.math.order(a, b);
        }
    }.order;
}

/// Returns a comparison function for descending order of any orderable type.
///
/// Time: O(1) | Space: O(1)
pub fn descending(comptime T: type) fn (ctx: void, a: T, b: T) std.math.Order {
    return struct {
        pub fn order(_: void, a: T, b: T) std.math.Order {
            return std.math.order(a, b).invert();
        }
    }.order;
}

/// Returns a comparison function for string comparison (lexicographic order).
///
/// Time: O(min(|a|, |b|)) | Space: O(1)
pub fn stringAscending(_: void, a: []const u8, b: []const u8) std.math.Order {
    return std.mem.order(u8, a, b);
}

/// Returns a comparison function for descending string order.
///
/// Time: O(min(|a|, |b|)) | Space: O(1)
pub fn stringDescending(_: void, a: []const u8, b: []const u8) std.math.Order {
    return std.mem.order(u8, a, b).invert();
}

/// Returns a comparison function that compares tuples by first element, then second, etc.
/// Useful for multi-column sorting.
///
/// Time: O(1) per field | Space: O(1)
pub fn tuple2(comptime T1: type, comptime T2: type) fn (ctx: void, a: struct { T1, T2 }, b: struct { T1, T2 }) std.math.Order {
    return struct {
        pub fn order(_: void, a: struct { T1, T2 }, b: struct { T1, T2 }) std.math.Order {
            const first = std.math.order(a[0], b[0]);
            if (first != .eq) return first;
            return std.math.order(a[1], b[1]);
        }
    }.order;
}

/// Returns a comparison function for pointers (compares pointed values).
/// Useful when keys are pointers but you want value-based comparison.
///
/// Time: O(1) | Space: O(1)
pub fn deref(comptime T: type) fn (ctx: void, a: *const T, b: *const T) std.math.Order {
    return struct {
        pub fn order(_: void, a: *const T, b: *const T) std.math.Order {
            return std.math.order(a.*, b.*);
        }
    }.order;
}

// Tests

const testing = std.testing;

test "ascending i64" {
    const cmp = ascending(i64);
    try testing.expectEqual(.lt, cmp({}, -5, 10));
    try testing.expectEqual(.eq, cmp({}, 42, 42));
    try testing.expectEqual(.gt, cmp({}, 100, -1));
}

test "descending i64" {
    const cmp = descending(i64);
    try testing.expectEqual(.gt, cmp({}, -5, 10));
    try testing.expectEqual(.eq, cmp({}, 42, 42));
    try testing.expectEqual(.lt, cmp({}, 100, -1));
}

test "ascending f64" {
    const cmp = ascending(f64);
    try testing.expectEqual(.lt, cmp({}, 1.5, 2.5));
    try testing.expectEqual(.eq, cmp({}, 3.14, 3.14));
    try testing.expectEqual(.gt, cmp({}, 10.0, 9.9));
}

test "stringAscending" {
    try testing.expectEqual(.lt, stringAscending({}, "apple", "banana"));
    try testing.expectEqual(.eq, stringAscending({}, "test", "test"));
    try testing.expectEqual(.gt, stringAscending({}, "zebra", "ant"));
    try testing.expectEqual(.lt, stringAscending({}, "a", "aa")); // shorter comes first
}

test "stringDescending" {
    try testing.expectEqual(.gt, stringDescending({}, "apple", "banana"));
    try testing.expectEqual(.eq, stringDescending({}, "test", "test"));
    try testing.expectEqual(.lt, stringDescending({}, "zebra", "ant"));
}

test "tuple2 comparison" {
    const cmp = tuple2(i32, i32);
    const Tuple = struct { i32, i32 };

    try testing.expectEqual(.lt, cmp({}, Tuple{ 1, 2 }, Tuple{ 2, 1 })); // first differs
    try testing.expectEqual(.lt, cmp({}, Tuple{ 1, 2 }, Tuple{ 1, 3 })); // first same, second differs
    try testing.expectEqual(.eq, cmp({}, Tuple{ 5, 10 }, Tuple{ 5, 10 })); // both same
    try testing.expectEqual(.gt, cmp({}, Tuple{ 2, 1 }, Tuple{ 1, 9 })); // first differs (takes precedence)
}

test "deref comparison" {
    const cmp = deref(i32);
    const a: i32 = 10;
    const b: i32 = 20;
    const c: i32 = 10;

    try testing.expectEqual(.lt, cmp({}, &a, &b));
    try testing.expectEqual(.eq, cmp({}, &a, &c));
    try testing.expectEqual(.gt, cmp({}, &b, &a));
}

test "integration with RedBlackTree" {
    const RBTree = @import("../containers/trees/red_black_tree.zig").RedBlackTree;

    var tree = RBTree(i64, []const u8, void, ascending(i64)).init(testing.allocator, {});
    defer tree.deinit();

    _ = try tree.insert(5, "five");
    _ = try tree.insert(2, "two");
    _ = try tree.insert(8, "eight");

    try testing.expectEqual(@as(?[]const u8, "two"), tree.get(2));
    try testing.expectEqual(@as(?[]const u8, "five"), tree.get(5));
    try testing.expectEqual(@as(?[]const u8, "eight"), tree.get(8));
    try testing.expectEqual(@as(usize, 3), tree.count());
}
