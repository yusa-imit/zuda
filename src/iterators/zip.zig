//! Zip Iterator Adaptor
//!
//! The Zip adaptor combines two iterators element-wise into tuples containing
//! an element from each iterator. The resulting iterator stops when either
//! input iterator is exhausted, ensuring paired element access.
//!
//! ## Type Parameters
//! - T: Element type of first iterator
//! - U: Element type of second iterator
//! - FirstIter: First iterator type (must have next() -> ?T)
//! - SecondIter: Second iterator type (must have next() -> ?U)
//!
//! ## Return Type
//! Each call to next() returns `?struct { a: T, b: U }` where:
//! - `a` is the element from the first iterator
//! - `b` is the element from the second iterator
//!
//! ## Time Complexity
//! - next(): O(1) — single method call on each iterator
//!
//! ## Space Complexity
//! - O(1) — stores only two iterator instances, no additional allocations
//!
//! ## Example
//! ```zig
//! var first = [_]i32{1, 2, 3};
//! var second = [_]f32{1.5, 2.5, 3.5};
//! var first_iter = SliceIterator(i32).init(&first);
//! var second_iter = SliceIterator(f32).init(&second);
//! var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32))
//!     .init(&first_iter, &second_iter);
//! while (zipped.next()) |pair| {
//!     // pair.a is i32, pair.b is f32
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Zip adaptor that combines two iterators element-wise into tuples.
/// Time: O(1) per element
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes two concrete iterator types
/// and returns a struct that zips them together into tuples.
pub fn Zip(comptime T: type, comptime U: type, comptime FirstIter: type, comptime SecondIter: type) type {
    return struct {
        const Self = @This();

        /// Pointer to first iterator
        first_iter: *FirstIter,
        /// Pointer to second iterator
        second_iter: *SecondIter,

        /// Initialize the Zip adaptor with two iterators.
        /// Time: O(1) | Space: O(1)
        pub fn init(first_iter: *FirstIter, second_iter: *SecondIter) Self {
            return .{
                .first_iter = first_iter,
                .second_iter = second_iter,
            };
        }

        /// Get the next pair of elements from the zip.
        /// Returns null when either iterator is exhausted.
        /// Time: O(1) | Space: O(1)
        pub fn next(self: *Self) ?struct { a: T, b: U } {
            const first_val = self.first_iter.next() orelse return null;
            const second_val = self.second_iter.next() orelse return null;
            return .{ .a = first_val, .b = second_val };
        }
    };
}

// -- Tests --

/// Simple slice iterator for testing (not part of public API)
fn SliceIterator(comptime T: type) type {
    return struct {
        const SelfIter = @This();

        slice: []const T,
        index: usize = 0,

        fn init(slice: []const T) SelfIter {
            return .{ .slice = slice };
        }

        fn next(self: *SelfIter) ?T {
            if (self.index >= self.slice.len) return null;
            defer self.index += 1;
            return self.slice[self.index];
        }
    };
}

test "zip: basic zipping of two equal-length iterators" {
    var first = [_]i32{ 1, 2, 3 };
    var second = [_]f32{ 1.5, 2.5, 3.5 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expect(pair1 != null);
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(1.5, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expect(pair2 != null);
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(2.5, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expect(pair3 != null);
    try testing.expectEqual(3, pair3.?.a);
    try testing.expectEqual(3.5, pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: both iterators same type (i32, i32)" {
    var first = [_]i32{ 10, 20, 30 };
    var second = [_]i32{ 100, 200, 300 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(10, pair1.?.a);
    try testing.expectEqual(100, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(20, pair2.?.a);
    try testing.expectEqual(200, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(30, pair3.?.a);
    try testing.expectEqual(300, pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: first iterator shorter than second" {
    var first = [_]i32{ 1, 2 };
    var second = [_]i32{ 10, 20, 30, 40 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expect(pair1 != null);
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(10, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expect(pair2 != null);
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(20, pair2.?.b);

    // Should stop after first iterator exhausted
    try testing.expectEqual(null, zipped.next());
}

test "zip: second iterator shorter than first" {
    var first = [_]i32{ 1, 2, 3, 4 };
    var second = [_]i32{ 10, 20 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expect(pair1 != null);
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(10, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expect(pair2 != null);
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(20, pair2.?.b);

    // Should stop after second iterator exhausted
    try testing.expectEqual(null, zipped.next());
}

test "zip: both iterators empty" {
    var first = [_]i32{};
    var second = [_]f32{};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    try testing.expectEqual(null, zipped.next());
    try testing.expectEqual(null, zipped.next());
}

test "zip: first iterator empty, second non-empty" {
    var first = [_]i32{};
    var second = [_]f32{ 1.5, 2.5, 3.5 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    try testing.expectEqual(null, zipped.next());
}

test "zip: first iterator non-empty, second empty" {
    var first = [_]i32{ 1, 2, 3 };
    var second = [_]f32{};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    try testing.expectEqual(null, zipped.next());
}

test "zip: single element from each iterator" {
    var first = [_]i32{42};
    var second = [_]f32{3.14};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    const pair = zipped.next();
    try testing.expect(pair != null);
    try testing.expectEqual(42, pair.?.a);
    try testing.expectEqual(3.14, pair.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: mixed numeric types (i32 and f32)" {
    var first = [_]i32{ 1, 2, 3 };
    var second = [_]f32{ 10.5, 20.5, 30.5 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(10.5, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(20.5, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(3, pair3.?.a);
    try testing.expectEqual(30.5, pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: mixed numeric types (i32 and u32)" {
    var first = [_]i32{ 1, 2, 3 };
    var second = [_]u32{ 10, 20, 30 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(u32).init(&second);
    var zipped = Zip(i32, u32, SliceIterator(i32), SliceIterator(u32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(10, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(20, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(3, pair3.?.a);
    try testing.expectEqual(30, pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: integer and boolean types" {
    var first = [_]i32{ 1, 2, 3 };
    var second = [_]bool{ true, false, true };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(bool).init(&second);
    var zipped = Zip(i32, bool, SliceIterator(i32), SliceIterator(bool)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(1, pair1.?.a);
    try testing.expectEqual(true, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(2, pair2.?.a);
    try testing.expectEqual(false, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(3, pair3.?.a);
    try testing.expectEqual(true, pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: exhaustion requires multiple next calls" {
    var first = [_]i32{1};
    var second = [_]f32{1.5};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    const pair = zipped.next();
    try testing.expect(pair != null);
    try testing.expectEqual(1, pair.?.a);

    try testing.expectEqual(null, zipped.next());
    try testing.expectEqual(null, zipped.next());
    try testing.expectEqual(null, zipped.next());
}

test "zip: large dataset with 1000 elements" {
    var first: [1000]i32 = undefined;
    var second: [1000]i32 = undefined;

    for (0..1000) |i| {
        first[i] = @intCast(i);
        second[i] = @intCast(i * 10);
    }

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    for (0..1000) |i| {
        const pair = zipped.next();
        try testing.expect(pair != null);
        try testing.expectEqual(@as(i32, @intCast(i)), pair.?.a);
        try testing.expectEqual(@as(i32, @intCast(i * 10)), pair.?.b);
    }

    try testing.expectEqual(null, zipped.next());
}

test "zip: state isolation between independent zip instances" {
    var first1 = [_]i32{ 1, 2, 3 };
    var second1 = [_]i32{ 10, 20, 30 };

    var first2 = [_]i32{ 100, 200, 300 };
    var second2 = [_]i32{ 1000, 2000, 3000 };

    var first_iter1 = SliceIterator(i32).init(&first1);
    var second_iter1 = SliceIterator(i32).init(&second1);
    var zipped1 = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter1, &second_iter1);

    var first_iter2 = SliceIterator(i32).init(&first2);
    var second_iter2 = SliceIterator(i32).init(&second2);
    var zipped2 = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter2, &second_iter2);

    const pair1_1 = zipped1.next();
    try testing.expectEqual(1, pair1_1.?.a);
    try testing.expectEqual(10, pair1_1.?.b);

    const pair2_1 = zipped2.next();
    try testing.expectEqual(100, pair2_1.?.a);
    try testing.expectEqual(1000, pair2_1.?.b);

    const pair1_2 = zipped1.next();
    try testing.expectEqual(2, pair1_2.?.a);
    try testing.expectEqual(20, pair1_2.?.b);

    const pair2_2 = zipped2.next();
    try testing.expectEqual(200, pair2_2.?.a);
    try testing.expectEqual(2000, pair2_2.?.b);
}

test "zip: negative numbers" {
    var first = [_]i32{ -5, -3, 0, 3, 5 };
    var second = [_]i32{ -10, -6, 0, 6, 10 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(-5, pair1.?.a);
    try testing.expectEqual(-10, pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(-3, pair2.?.a);
    try testing.expectEqual(-6, pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(0, pair3.?.a);
    try testing.expectEqual(0, pair3.?.b);

    const pair4 = zipped.next();
    try testing.expectEqual(3, pair4.?.a);
    try testing.expectEqual(6, pair4.?.b);

    const pair5 = zipped.next();
    try testing.expectEqual(5, pair5.?.a);
    try testing.expectEqual(10, pair5.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: zero values in both positions" {
    var first = [_]i32{ 0, 0, 0 };
    var second = [_]i32{ 0, 0, 0 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    for (0..3) |_| {
        const pair = zipped.next();
        try testing.expectEqual(0, pair.?.a);
        try testing.expectEqual(0, pair.?.b);
    }

    try testing.expectEqual(null, zipped.next());
}

test "zip: maximum values handling" {
    const max_i32 = std.math.maxInt(i32);
    const max_u32 = std.math.maxInt(u32);

    var first = [_]i32{max_i32};
    var second = [_]u32{max_u32};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(u32).init(&second);
    var zipped = Zip(i32, u32, SliceIterator(i32), SliceIterator(u32)).init(&first_iter, &second_iter);

    const pair = zipped.next();
    try testing.expectEqual(max_i32, pair.?.a);
    try testing.expectEqual(max_u32, pair.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: minimum values handling" {
    const min_i32 = std.math.minInt(i32);

    var first = [_]i32{min_i32};
    var second = [_]i32{0};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    const pair = zipped.next();
    try testing.expectEqual(min_i32, pair.?.a);
    try testing.expectEqual(0, pair.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: float values with precision" {
    var first = [_]f32{ 1.5, 2.7, 3.14 };
    var second = [_]f32{ 10.5, 20.7, 30.14 };

    var first_iter = SliceIterator(f32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(f32, f32, SliceIterator(f32), SliceIterator(f32)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectApproxEqAbs(1.5, pair1.?.a, 0.01);
    try testing.expectApproxEqAbs(10.5, pair1.?.b, 0.01);

    const pair2 = zipped.next();
    try testing.expectApproxEqAbs(2.7, pair2.?.a, 0.01);
    try testing.expectApproxEqAbs(20.7, pair2.?.b, 0.01);

    const pair3 = zipped.next();
    try testing.expectApproxEqAbs(3.14, pair3.?.a, 0.01);
    try testing.expectApproxEqAbs(30.14, pair3.?.b, 0.01);

    try testing.expectEqual(null, zipped.next());
}

test "zip: struct fields access" {
    const Point = struct { x: i32, y: i32 };
    var first = [_]Point{ .{ .x = 1, .y = 2 }, .{ .x = 3, .y = 4 } };
    var second = [_]Point{ .{ .x = 10, .y = 20 }, .{ .x = 30, .y = 40 } };

    var first_iter = SliceIterator(Point).init(&first);
    var second_iter = SliceIterator(Point).init(&second);
    var zipped = Zip(Point, Point, SliceIterator(Point), SliceIterator(Point)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(1, pair1.?.a.x);
    try testing.expectEqual(2, pair1.?.a.y);
    try testing.expectEqual(10, pair1.?.b.x);
    try testing.expectEqual(20, pair1.?.b.y);

    const pair2 = zipped.next();
    try testing.expectEqual(3, pair2.?.a.x);
    try testing.expectEqual(4, pair2.?.a.y);
    try testing.expectEqual(30, pair2.?.b.x);
    try testing.expectEqual(40, pair2.?.b.y);

    try testing.expectEqual(null, zipped.next());
}

test "zip: alternating access pattern" {
    var first = [_]i32{ 1, 2, 3, 4, 5 };
    var second = [_]i32{ 10, 20, 30, 40, 50 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    var results: [5]struct { a: i32, b: i32 } = undefined;
    var idx: usize = 0;

    while (zipped.next()) |pair| {
        results[idx] = pair;
        idx += 1;
    }

    try testing.expectEqual(5, idx);
    try testing.expectEqual(1, results[0].a);
    try testing.expectEqual(10, results[0].b);
    try testing.expectEqual(5, results[4].a);
    try testing.expectEqual(50, results[4].b);
}

test "zip: different sized types (u8 and u64)" {
    var first = [_]u8{ 1, 2, 3 };
    var second = [_]u64{ 100, 200, 300 };

    var first_iter = SliceIterator(u8).init(&first);
    var second_iter = SliceIterator(u64).init(&second);
    var zipped = Zip(u8, u64, SliceIterator(u8), SliceIterator(u64)).init(&first_iter, &second_iter);

    const pair1 = zipped.next();
    try testing.expectEqual(@as(u8, 1), pair1.?.a);
    try testing.expectEqual(@as(u64, 100), pair1.?.b);

    const pair2 = zipped.next();
    try testing.expectEqual(@as(u8, 2), pair2.?.a);
    try testing.expectEqual(@as(u64, 200), pair2.?.b);

    const pair3 = zipped.next();
    try testing.expectEqual(@as(u8, 3), pair3.?.a);
    try testing.expectEqual(@as(u64, 300), pair3.?.b);

    try testing.expectEqual(null, zipped.next());
}

test "zip: pair structure preserves type information" {
    var first = [_]i32{42};
    var second = [_]bool{true};

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(bool).init(&second);
    var zipped = Zip(i32, bool, SliceIterator(i32), SliceIterator(bool)).init(&first_iter, &second_iter);

    const pair = zipped.next();
    try testing.expect(pair != null);

    // Access fields to verify types
    const a_val: i32 = pair.?.a;
    const b_val: bool = pair.?.b;

    try testing.expectEqual(42, a_val);
    try testing.expectEqual(true, b_val);
}

test "zip: long chains with mixed types" {
    var first = [_]i32{ 1, 2, 3, 4, 5 };
    var second = [_]f32{ 1.1, 2.2, 3.3, 4.4, 5.5 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(f32).init(&second);
    var zipped = Zip(i32, f32, SliceIterator(i32), SliceIterator(f32)).init(&first_iter, &second_iter);

    var count: usize = 0;
    while (zipped.next()) |pair| {
        count += 1;
        // Verify a is always less than b when cast to f32
        const a_as_float: f32 = @floatFromInt(pair.a);
        try testing.expect(a_as_float < pair.b);
    }

    try testing.expectEqual(5, count);
}

test "zip: all pairs consumed, state verified exhausted" {
    var first = [_]i32{ 10, 20 };
    var second = [_]i32{ 100, 200 };

    var first_iter = SliceIterator(i32).init(&first);
    var second_iter = SliceIterator(i32).init(&second);
    var zipped = Zip(i32, i32, SliceIterator(i32), SliceIterator(i32)).init(&first_iter, &second_iter);

    // Consume all pairs
    _ = zipped.next();
    _ = zipped.next();

    // Verify exhaustion
    try testing.expectEqual(null, zipped.next());
    try testing.expectEqual(null, zipped.next());
    try testing.expectEqual(null, zipped.next());
}
