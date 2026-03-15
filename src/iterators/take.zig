//! Take Iterator Adaptor
//!
//! The Take adaptor limits iteration to the first N elements from a base iterator.
//! Once N elements have been yielded, the iterator returns null for all subsequent
//! calls, even if the base iterator has more elements.
//!
//! ## Type Parameters
//! - T: Element type
//! - BaseIter: The underlying iterator type
//!
//! ## Time Complexity
//! - next(): O(1) — delegates to base iterator
//!
//! ## Space Complexity
//! - O(1) — no allocation; stores only base iterator, count, and remaining limit
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var take = Take(i32, SliceIterator(i32)).init(slice_iter, 3);
//! while (take.next()) |val| {
//!     // val will be 1, 2, 3 (then stop)
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Take adaptor that limits iteration to the first N elements.
/// Time: O(1) per element
/// Space: O(1) — no additional allocations
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn Take(comptime T: type, comptime BaseIter: type) type {
    // Determine if BaseIter is a pointer type
    const base_iter_info = @typeInfo(BaseIter);
    const is_pointer = switch (base_iter_info) {
        .pointer => true,
        else => false,
    };

    // If BaseIter is not a pointer, create a struct that can accept either
    // value or pointer through type flexibility
    if (!is_pointer) {
        return struct {
            const Self = @This();

            /// Base iterator (concrete type with next() -> ?T method)
            base_iter: BaseIter,
            /// Number of elements remaining to yield
            remaining: usize,

            /// Initialize the Take adaptor with a base iterator and count.
            /// Can accept both value and pointer types for base_iter.
            /// Time: O(1) | Space: O(1)
            pub fn init(base_iter: anytype, n: usize) Self {
                const input_type = @TypeOf(base_iter);
                const input_is_ptr = switch (@typeInfo(input_type)) {
                    .pointer => true,
                    else => false,
                };

                // If input is a pointer to BaseIter, we can reinterpret it
                const actual_value: BaseIter = if (input_is_ptr)
                    // Cast the pointer value itself as the struct instance
                    // This works if BaseIter has the same memory layout as a pointer
                    @as(*BaseIter, @ptrCast(base_iter)).*
                else
                    base_iter;

                return .{
                    .base_iter = actual_value,
                    .remaining = n,
                };
            }

            /// Get the next element if remaining count > 0.
            /// Once N elements have been yielded, always returns null.
            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Self) ?T {
                if (self.remaining == 0) return null;
                const value = self.base_iter.next() orelse return null;
                self.remaining -= 1;
                return value;
            }
        };
    } else {
        // BaseIter is a pointer
        return struct {
            const Self = @This();

            /// Base iterator (pointer type with next() -> ?T method)
            base_iter: BaseIter,
            /// Number of elements remaining to yield
            remaining: usize,

            /// Initialize the Take adaptor with a base iterator and count.
            /// Time: O(1) | Space: O(1)
            pub fn init(base_iter: BaseIter, n: usize) Self {
                return .{
                    .base_iter = base_iter,
                    .remaining = n,
                };
            }

            /// Get the next element if remaining count > 0.
            /// Once N elements have been yielded, always returns null.
            /// Time: O(1) | Space: O(1)
            pub fn next(self: *Self) ?T {
                if (self.remaining == 0) return null;
                const value = self.base_iter.*.next() orelse return null;
                self.remaining -= 1;
                return value;
            }
        };
    }
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

// -- Unit Tests (18+ cases) --

test "take: basic take N < iterator length" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 3);

    try testing.expectEqual(1, take_it.next());
    try testing.expectEqual(2, take_it.next());
    try testing.expectEqual(3, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: take exact length N == iterator length" {
    var numbers = [_]i32{ 10, 20, 30 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 3);

    try testing.expectEqual(10, take_it.next());
    try testing.expectEqual(20, take_it.next());
    try testing.expectEqual(30, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: take more than available N > iterator length" {
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 10);

    try testing.expectEqual(1, take_it.next());
    try testing.expectEqual(2, take_it.next());
    try testing.expectEqual(3, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: zero elements N = 0" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 0);

    try testing.expectEqual(null, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: empty iterator with N > 0" {
    var numbers = [_]i32{};
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 5);

    try testing.expectEqual(null, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: single element with N = 1" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(42, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: single element with N > 1" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 5);

    try testing.expectEqual(42, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: large dataset take 100 from 1000" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 100);

    for (0..100) |i| {
        const expected: i32 = @intCast(i);
        try testing.expectEqual(expected, take_it.next());
    }
    try testing.expectEqual(null, take_it.next());
}

test "take: take all elements N = total length" {
    var numbers = [_]i32{ 5, 10, 15, 20 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 4);

    try testing.expectEqual(5, take_it.next());
    try testing.expectEqual(10, take_it.next());
    try testing.expectEqual(15, take_it.next());
    try testing.expectEqual(20, take_it.next());
    try testing.expectEqual(null, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: iterator exhaustion with multiple next calls after exhausted" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 2);

    try testing.expectEqual(1, take_it.next());
    try testing.expectEqual(2, take_it.next());
    try testing.expectEqual(null, take_it.next());
    try testing.expectEqual(null, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: state isolation independent take instances" {
    var numbers1 = [_]i32{ 1, 2, 3, 4, 5 };
    var numbers2 = [_]i32{ 10, 20, 30, 40, 50 };

    const iter1 = SliceIterator(i32).init(&numbers1);
    var take1 = Take(i32, SliceIterator(i32)).init(iter1, 2);

    const iter2 = SliceIterator(i32).init(&numbers2);
    var take2 = Take(i32, SliceIterator(i32)).init(iter2, 3);

    try testing.expectEqual(1, take1.next());
    try testing.expectEqual(10, take2.next());
    try testing.expectEqual(2, take1.next());
    try testing.expectEqual(20, take2.next());
    try testing.expectEqual(null, take1.next());
    try testing.expectEqual(30, take2.next());
    try testing.expectEqual(null, take2.next());
}

test "take: with f32 type" {
    var numbers = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const iter = SliceIterator(f32).init(&numbers);
    var take_it = Take(f32, SliceIterator(f32)).init(iter, 2);

    try testing.expectEqual(1.5, take_it.next());
    try testing.expectEqual(2.5, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: with bool type" {
    var numbers = [_]bool{ true, false, true, false, true };
    const iter = SliceIterator(bool).init(&numbers);
    var take_it = Take(bool, SliceIterator(bool)).init(iter, 3);

    try testing.expectEqual(true, take_it.next());
    try testing.expectEqual(false, take_it.next());
    try testing.expectEqual(true, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: with struct type" {
    const Point = struct { x: i32, y: i32 };
    var points = [_]Point{
        .{ .x = 1, .y = 2 },
        .{ .x = 3, .y = 4 },
        .{ .x = 5, .y = 6 },
    };
    const iter = SliceIterator(Point).init(&points);
    var take_it = Take(Point, SliceIterator(Point)).init(iter, 2);

    const p1 = take_it.next();
    try testing.expect(p1 != null);
    try testing.expectEqual(1, p1.?.x);
    try testing.expectEqual(2, p1.?.y);

    const p2 = take_it.next();
    try testing.expect(p2 != null);
    try testing.expectEqual(3, p2.?.x);
    try testing.expectEqual(4, p2.?.y);

    try testing.expectEqual(null, take_it.next());
}

test "take: boundary condition max i32 value" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{ max_i32, 42, 100 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(max_i32, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: boundary condition min i32 value" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{ min_i32, 42, 100 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 1);

    try testing.expectEqual(min_i32, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: chained takes take(take(...))" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var take1 = Take(i32, SliceIterator(i32)).init(iter, 4);
    var take2 = Take(i32, @TypeOf(take1)).init(&take1, 2);

    // take1 limits to first 4: [1,2,3,4]
    // take2 limits to first 2 of those: [1,2]
    try testing.expectEqual(1, take2.next());
    try testing.expectEqual(2, take2.next());
    try testing.expectEqual(null, take2.next());
}

test "take: three-level chained takes" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var take1 = Take(i32, SliceIterator(i32)).init(iter, 7);
    var take2 = Take(i32, @TypeOf(take1)).init(&take1, 5);
    var take3 = Take(i32, @TypeOf(take2)).init(&take2, 2);

    // take1: [1,2,3,4,5,6,7]
    // take2: [1,2,3,4,5]
    // take3: [1,2]
    try testing.expectEqual(1, take3.next());
    try testing.expectEqual(2, take3.next());
    try testing.expectEqual(null, take3.next());
}

test "take: negative concept not applicable usize is unsigned" {
    // usize cannot hold negative values, so this test verifies
    // that take with a very large usize value works correctly
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    const large_n: usize = 9999999;
    var take_it = Take(i32, SliceIterator(i32)).init(iter, large_n);

    try testing.expectEqual(1, take_it.next());
    try testing.expectEqual(2, take_it.next());
    try testing.expectEqual(3, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: zero value in iterator" {
    var numbers = [_]i32{ 0, 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 2);

    try testing.expectEqual(0, take_it.next());
    try testing.expectEqual(1, take_it.next());
    try testing.expectEqual(null, take_it.next());
}

test "take: alternating take pattern simulation" {
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var take_it = Take(i32, SliceIterator(i32)).init(iter, 5);

    var count: usize = 0;
    while (take_it.next()) |val| {
        count += 1;
        try testing.expect(val >= 1 and val <= 5);
    }
    try testing.expectEqual(5, count);
}
