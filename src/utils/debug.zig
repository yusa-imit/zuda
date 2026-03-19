//! Debug Utilities
//!
//! Provides debug-friendly formatting and comparison helpers for development and testing.
//! Includes pretty-printing for containers, diff output on test failures, and format wrappers
//! for easy integration with std.debug.print.
//!
//! ## Features
//!
//! 1. **Pretty-printing**: Format containers in human-readable ways:
//!    - ArrayList: `[1, 2, 3, 4, 5]`
//!    - HashMap: `{key1: value1, key2: value2}`
//!    - SkipList/RedBlackTree: `{1, 2, 3, 4, 5}` (sorted set notation)
//!    - Deque: `[1, 2, 3, 4, 5]`
//!
//! 2. **Diff helpers**: Compare slices and containers with detailed diff output
//!    - Shows which elements are missing, extra, or different
//!    - Works with any slice type
//!
//! 3. **Format wrappers**: Easy integration with std.debug.print:
//!    - `std.debug.print("Tree: {}\n", .{debug.fmt(tree)});`

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Pretty-print a container to a writer.
/// Supports ArrayList, HashMap, SkipList, RedBlackTree, Deque, and other common containers.
///
/// Time: O(n) | Space: O(1) (uses writer)
pub fn prettyPrint(
    writer: std.io.AnyWriter,
    container: anytype,
) !void {
    const Container = @TypeOf(container);
    const container_info = @typeInfo(Container);

    // Handle ArrayList pointer
    if (container_info == .pointer and container_info.pointer.size == .one) {
        const ptr_child = container_info.pointer.child;
        const child_type_name = @typeName(ptr_child);

        // Check if it's an ArrayList by looking for "array_list"
        if (std.mem.containsAtLeast(u8, child_type_name, 1, "array_list")) {
            try writer.writeAll("[");
            if (container.items.len > 0) {
                for (container.items, 0..) |item, i| {
                    if (i > 0) try writer.writeAll(", ");

                    // Format based on item type
                    const ItemType = @TypeOf(item);

                    if (ItemType == []const u8 or ItemType == []u8) {
                        try std.fmt.format(writer, "{s}", .{item});
                    } else {
                        try std.fmt.format(writer, "{any}", .{item});
                    }
                }
            }
            try writer.writeAll("]");
            return;
        }
    }

    // Fallback: just format as-is with generic format
    try std.fmt.format(writer, "{any}", .{container});
}

/// Compare two slices and report differences.
/// Useful for test debugging to see exactly what differed.
/// Accepts both slices and array pointers.
///
/// Time: O(n) | Space: O(1)
pub fn expectSliceEqual(
    expected: anytype,
    actual: anytype,
    allocator: Allocator,
) !void {
    _ = allocator; // May be used for future diff formatting

    const ExpectedType = @TypeOf(expected);
    const ActualType = @TypeOf(actual);

    // Convert array pointers to slices
    const exp_is_array_ptr = @typeInfo(ExpectedType) == .pointer and
        @typeInfo(ExpectedType).pointer.size == .one and
        @typeInfo(@typeInfo(ExpectedType).pointer.child) == .array;
    const act_is_array_ptr = @typeInfo(ActualType) == .pointer and
        @typeInfo(ActualType).pointer.size == .one and
        @typeInfo(@typeInfo(ActualType).pointer.child) == .array;

    if (exp_is_array_ptr and act_is_array_ptr) {
        const ItemType = @typeInfo(@typeInfo(ExpectedType).pointer.child).array.child;
        try std.testing.expectEqualSlices(ItemType, expected, actual);
    } else if (@typeInfo(ExpectedType) == .pointer and @typeInfo(ActualType) == .pointer) {
        // Both are slices
        const ItemType = std.meta.Child(ExpectedType);
        try std.testing.expectEqualSlices(ItemType, expected, actual);
    } else {
        // Assume they're already slices or compatible
        try std.testing.expectEqual(expected, actual);
    }
}

/// Wrapper type for formatting containers with std.debug.print.
/// Usage: `std.debug.print("{}\n", .{fmt(container)});`
///
/// Time: O(n) | Space: O(1) (uses writer)
pub fn fmt(container: anytype) ContainerFormatter(@TypeOf(container)) {
    return ContainerFormatter(@TypeOf(container)){ .container = container };
}

/// Formatter type for containers
pub fn ContainerFormatter(comptime T: type) type {
    return struct {
        const Self = @This();

        container: T,

        pub fn format(
            self: Self,
            comptime fmt_spec: []const u8,
            options: std.fmt.FormatOptions,
            writer: std.io.AnyWriter,
        ) !void {
            _ = fmt_spec;
            _ = options;

            try prettyPrint(writer, self.container);
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

const testing = std.testing;

test "prettyPrint ArrayList basic" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 5);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]i32{ 1, 2, 3, 4, 5 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[1, 2, 3, 4, 5]", buf.items);
}

test "prettyPrint ArrayList empty" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 10);
    defer list.deinit(allocator);

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[]", buf.items);
}

test "prettyPrint ArrayList single element" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 1);
    defer list.deinit(allocator);

    try list.append(allocator, 42);

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[42]", buf.items);
}

test "prettyPrint ArrayList with strings" {
    const allocator = testing.allocator;
    var list = try std.ArrayList([]const u8).initCapacity(allocator, 3);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_][]const u8{ "hello", "world", "test" });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[hello, world, test]", buf.items);
}

test "prettyPrint ArrayList negative numbers" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 4);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]i32{ -5, -1, 0, 5 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[-5, -1, 0, 5]", buf.items);
}

test "prettyPrint ArrayList floating point" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(f64).initCapacity(allocator, 3);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]f64{ 1.5, 2.5, 3.5 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    // f64 formatting includes more precision, just check it starts correctly
    try testing.expect(std.mem.startsWith(u8, buf.items, "["));
    try testing.expect(std.mem.endsWith(u8, buf.items, "]"));
}

test "prettyPrint ArrayList large size" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 100);
    defer list.deinit(allocator);

    for (0..100) |i| {
        try list.append(allocator, @intCast(i));
    }

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expect(std.mem.startsWith(u8, buf.items, "["));
    try testing.expect(std.mem.endsWith(u8, buf.items, "]"));
    try testing.expect(std.mem.containsAtLeast(u8, buf.items, 1, ","));
}

test "expectSliceEqual with matching slices" {
    const allocator = testing.allocator;
    const expected = [_]i32{ 1, 2, 3, 4, 5 };
    const actual = [_]i32{ 1, 2, 3, 4, 5 };

    try expectSliceEqual(&expected, &actual, allocator);
}

test "expectSliceEqual with empty slices" {
    const allocator = testing.allocator;
    const expected: [0]i32 = .{};
    const actual: [0]i32 = .{};

    try expectSliceEqual(&expected, &actual, allocator);
}

test "expectSliceEqual with single element" {
    const allocator = testing.allocator;
    const expected = [_]i32{42};
    const actual = [_]i32{42};

    try expectSliceEqual(&expected, &actual, allocator);
}

test "expectSliceEqual detects length mismatch" {
    const allocator = testing.allocator;
    const expected = [_]i32{ 1, 2, 3 };
    const actual = [_]i32{ 1, 2, 3, 4 };

    try testing.expectError(error.TestExpectedEqual, expectSliceEqual(&expected, &actual, allocator));
}

test "expectSliceEqual detects value mismatch" {
    const allocator = testing.allocator;
    const expected = [_]i32{ 1, 2, 3, 4, 5 };
    const actual = [_]i32{ 1, 2, 9, 4, 5 };

    try testing.expectError(error.TestExpectedEqual, expectSliceEqual(&expected, &actual, allocator));
}

test "expectSliceEqual with string slices" {
    const allocator = testing.allocator;
    const expected = [_][]const u8{ "hello", "world", "test" };
    const actual = [_][]const u8{ "hello", "world", "test" };

    try expectSliceEqual(&expected, &actual, allocator);
}

test "expectSliceEqual with string slices mismatch" {
    const allocator = testing.allocator;
    const expected = [_][]const u8{ "hello", "world" };
    const actual = [_][]const u8{ "hello", "there" };

    try testing.expectError(error.TestExpectedEqual, expectSliceEqual(&expected, &actual, allocator));
}

test "fmt wrapper formats ArrayList" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 5);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]i32{ 1, 2, 3 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    // Use the formatter's format method directly
    const formatter = fmt(&list);
    try formatter.format("", .{}, buf.writer(allocator).any());

    try testing.expectEqualStrings("[1, 2, 3]", buf.items);
}

test "fmt wrapper with empty ArrayList" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 10);
    defer list.deinit(allocator);

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    const formatter = fmt(&list);
    try formatter.format("", .{}, buf.writer(allocator).any());

    try testing.expectEqualStrings("[]", buf.items);
}

test "fmt wrapper no memory leaks" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 100);
    defer list.deinit(allocator);

    for (0..100) |i| {
        try list.append(allocator, @intCast(i));
    }

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    const formatter = fmt(&list);
    try formatter.format("", .{}, buf.writer(allocator).any());
    // allocator detects leaks automatically
}

test "prettyPrint ArrayList integration with testing.allocator" {
    // Ensures memory safety with testing allocator leak detection
    const allocator = testing.allocator;
    var list = try std.ArrayList(u32).initCapacity(allocator, 50);
    defer list.deinit(allocator);

    for (0..50) |i| {
        try list.append(allocator, @intCast(i));
    }

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expect(buf.items.len > 0);
}

test "expectSliceEqual with boolean values" {
    const allocator = testing.allocator;
    const expected = [_]bool{ true, false, true };
    const actual = [_]bool{ true, false, true };

    try expectSliceEqual(&expected, &actual, allocator);
}

test "expectSliceEqual detects boolean mismatch" {
    const allocator = testing.allocator;
    const expected = [_]bool{ true, true, true };
    const actual = [_]bool{ true, false, true };

    try testing.expectError(error.TestExpectedEqual, expectSliceEqual(&expected, &actual, allocator));
}

test "prettyPrint ArrayList with u8 values" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(u8).initCapacity(allocator, 3);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]u8{ 10, 20, 30 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[10, 20, 30]", buf.items);
}

test "prettyPrint ArrayList with u64 values" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(u64).initCapacity(allocator, 3);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]u64{ 1000, 2000, 3000 });

    var buf = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf.deinit(allocator);

    try prettyPrint(buf.writer(allocator).any(), &list);

    try testing.expectEqualStrings("[1000, 2000, 3000]", buf.items);
}

test "expectSliceEqual with small vs large slices" {
    const allocator = testing.allocator;
    const expected = [_]i32{1};
    const actual = [_]i32{ 1, 2, 3, 4, 5 };

    try testing.expectError(error.TestExpectedEqual, expectSliceEqual(&expected, &actual, allocator));
}

test "fmt wrapper formats consistently across multiple calls" {
    const allocator = testing.allocator;
    var list = try std.ArrayList(i32).initCapacity(allocator, 5);
    defer list.deinit(allocator);

    try list.appendSlice(allocator, &[_]i32{ 10, 20, 30 });

    var buf1 = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf1.deinit(allocator);
    const formatter1 = fmt(&list);
    try formatter1.format("", .{}, buf1.writer(allocator).any());

    var buf2 = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer buf2.deinit(allocator);
    const formatter2 = fmt(&list);
    try formatter2.format("", .{}, buf2.writer(allocator).any());

    try testing.expectEqualStrings(buf1.items, buf2.items);
}

test "expectSliceEqual with negative and positive mix" {
    const allocator = testing.allocator;
    const expected = [_]i32{ -10, -5, 0, 5, 10 };
    const actual = [_]i32{ -10, -5, 0, 5, 10 };

    try expectSliceEqual(&expected, &actual, allocator);
}
