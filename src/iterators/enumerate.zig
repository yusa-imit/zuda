//! Enumerate Iterator Adaptor
//!
//! The Enumerate adaptor yields (index, value) pairs for each element from a base
//! iterator, automatically tracking the current index starting from 0.
//!
//! ## Type Parameters
//! - T: Element type yielded by the base iterator
//! - BaseIter: The concrete base iterator type
//!
//! ## Time Complexity
//! - next(): O(1) — just forwards to base iterator and increments index
//!
//! ## Space Complexity
//! - O(1) — stores only base iterator and an index counter
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{10, 20, 30};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var enumerate = Enumerate(i32, SliceIterator(i32)).init(slice_iter);
//! while (enumerate.next()) |pair| {
//!     // pair.index: 0, 1, 2
//!     // pair.value: 10, 20, 30
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Enumerate adaptor that yields (index, value) pairs.
/// Time: O(1) per element
/// Space: O(1) — stores only base iterator and index counter
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing that type.
pub fn Enumerate(comptime T: type, comptime BaseIter: type) type {
    return struct {
        const Self = @This();

        /// The pair type returned by next()
        pub const Pair = struct {
            index: usize,
            value: T,
        };

        /// Base iterator (concrete type with next() -> ?T method)
        base_iter: BaseIter,
        /// Current enumeration index, incremented with each next() call
        index: usize = 0,

        /// Initialize the Enumerate adaptor with a base iterator.
        /// Time: O(1) | Space: O(1)
        pub fn init(base_iter: BaseIter) Self {
            return .{
                .base_iter = base_iter,
                .index = 0,
            };
        }

        /// Get the next (index, value) pair.
        /// Returns null when the base iterator is exhausted.
        /// Time: O(1) | Space: O(1)
        pub fn next(self: *Self) ?Pair {
            const value = self.base_iter.next() orelse return null;
            defer self.index += 1;
            return .{
                .index = self.index,
                .value = value,
            };
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

// -- Unit Tests (18+ cases) --

test "enumerate: basic enumeration with three elements" {
    var numbers = [_]i32{ 10, 20, 30 };
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair0 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair0.index);
    try testing.expectEqual(@as(i32, 10), pair0.value);

    const pair1 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 1), pair1.index);
    try testing.expectEqual(@as(i32, 20), pair1.value);

    const pair2 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 2), pair2.index);
    try testing.expectEqual(@as(i32, 30), pair2.value);

    try testing.expectEqual(null, enumerate.next());
}

test "enumerate: empty iterator returns null immediately" {
    var numbers = [_]i32{};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    try testing.expectEqual(null, enumerate.next());
    try testing.expectEqual(null, enumerate.next());
}

test "enumerate: single element has index 0" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair.index);
    try testing.expectEqual(@as(i32, 42), pair.value);

    try testing.expectEqual(null, enumerate.next());
}

test "enumerate: large dataset - verify indices and values" {
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(@as(i32, @intCast(i)) * 10);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    for (0..1000) |i| {
        const pair = enumerate.next().?;
        try testing.expectEqual(@as(usize, i), pair.index);
        try testing.expectEqual(@as(i32, @intCast(@as(i32, @intCast(i)) * 10)), pair.value);
    }

    try testing.expectEqual(null, enumerate.next());
}

test "enumerate: iterator exhaustion - multiple calls after end" {
    var numbers = [_]i32{1};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair.index);

    try testing.expectEqual(null, enumerate.next());
    try testing.expectEqual(null, enumerate.next());
    try testing.expectEqual(null, enumerate.next());
}

test "enumerate: state isolation - independent enumerate instances" {
    var numbers1 = [_]i32{ 100, 200 };
    var numbers2 = [_]i32{ 1000, 2000, 3000 };

    const iter1 = SliceIterator(i32).init(&numbers1);
    var enum1 = Enumerate(i32, SliceIterator(i32)).init(iter1);

    const iter2 = SliceIterator(i32).init(&numbers2);
    var enum2 = Enumerate(i32, SliceIterator(i32)).init(iter2);

    // Interleave calls to both enumerators
    const pair1_0 = enum1.next().?;
    try testing.expectEqual(@as(usize, 0), pair1_0.index);
    try testing.expectEqual(@as(i32, 100), pair1_0.value);

    const pair2_0 = enum2.next().?;
    try testing.expectEqual(@as(usize, 0), pair2_0.index);
    try testing.expectEqual(@as(i32, 1000), pair2_0.value);

    const pair1_1 = enum1.next().?;
    try testing.expectEqual(@as(usize, 1), pair1_1.index);
    try testing.expectEqual(@as(i32, 200), pair1_1.value);

    const pair2_1 = enum2.next().?;
    try testing.expectEqual(@as(usize, 1), pair2_1.index);
    try testing.expectEqual(@as(i32, 2000), pair2_1.value);

    const pair2_2 = enum2.next().?;
    try testing.expectEqual(@as(usize, 2), pair2_2.index);
    try testing.expectEqual(@as(i32, 3000), pair2_2.value);

    try testing.expectEqual(null, enum1.next());
    try testing.expectEqual(null, enum2.next());
}

test "enumerate: different types - f32" {
    var numbers = [_]f32{ 1.5, 2.5, 3.5 };
    const iter = SliceIterator(f32).init(&numbers);
    var enumerate = Enumerate(f32, SliceIterator(f32)).init(iter);

    const pair0 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair0.index);
    try testing.expectEqual(@as(f32, 1.5), pair0.value);

    const pair1 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 1), pair1.index);
    try testing.expectEqual(@as(f32, 2.5), pair1.value);

    const pair2 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 2), pair2.index);
    try testing.expectEqual(@as(f32, 3.5), pair2.value);
}

test "enumerate: different types - bool" {
    var bools = [_]bool{ true, false, true };
    const iter = SliceIterator(bool).init(&bools);
    var enumerate = Enumerate(bool, SliceIterator(bool)).init(iter);

    const pair0 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair0.index);
    try testing.expectEqual(true, pair0.value);

    const pair1 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 1), pair1.index);
    try testing.expectEqual(false, pair1.value);

    const pair2 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 2), pair2.index);
    try testing.expectEqual(true, pair2.value);
}

test "enumerate: different types - struct" {
    const Point = struct { x: i32, y: i32 };
    var points = [_]Point{
        .{ .x = 1, .y = 2 },
        .{ .x = 3, .y = 4 },
    };
    const iter = SliceIterator(Point).init(&points);
    var enumerate = Enumerate(Point, SliceIterator(Point)).init(iter);

    const pair0 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair0.index);
    try testing.expectEqual(@as(i32, 1), pair0.value.x);
    try testing.expectEqual(@as(i32, 2), pair0.value.y);

    const pair1 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 1), pair1.index);
    try testing.expectEqual(@as(i32, 3), pair1.value.x);
    try testing.expectEqual(@as(i32, 4), pair1.value.y);
}

test "enumerate: verify index increments correctly from 0 to 9" {
    var numbers: [10]i32 = undefined;
    for (0..10) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    for (0..10) |expected_index| {
        const pair = enumerate.next().?;
        try testing.expectEqual(@as(usize, expected_index), pair.index);
    }
}

test "enumerate: zero values in data" {
    var numbers = [_]i32{ 0, 0, 0 };
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair0 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair0.index);
    try testing.expectEqual(@as(i32, 0), pair0.value);

    const pair1 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 1), pair1.index);
    try testing.expectEqual(@as(i32, 0), pair1.value);

    const pair2 = enumerate.next().?;
    try testing.expectEqual(@as(usize, 2), pair2.index);
    try testing.expectEqual(@as(i32, 0), pair2.value);
}

test "enumerate: negative numbers with correct indices" {
    var numbers = [_]i32{ -5, -3, -1, 0, 1, 3, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const expected_values = [_]i32{ -5, -3, -1, 0, 1, 3, 5 };
    for (0..expected_values.len) |expected_index| {
        const pair = enumerate.next().?;
        try testing.expectEqual(@as(usize, expected_index), pair.index);
        try testing.expectEqual(expected_values[expected_index], pair.value);
    }
}

test "enumerate: max i32 value with correct index" {
    const max_i32 = std.math.maxInt(i32);
    var numbers = [_]i32{max_i32};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair.index);
    try testing.expectEqual(max_i32, pair.value);
}

test "enumerate: min i32 value with correct index" {
    const min_i32 = std.math.minInt(i32);
    var numbers = [_]i32{min_i32};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair = enumerate.next().?;
    try testing.expectEqual(@as(usize, 0), pair.index);
    try testing.expectEqual(min_i32, pair.value);
}

test "enumerate: index is independent of value magnitude" {
    var numbers = [_]i32{ 1000000, 2000000, 3000000 };
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    for (0..3) |expected_index| {
        const pair = enumerate.next().?;
        try testing.expectEqual(@as(usize, expected_index), pair.index);
    }
}

test "enumerate: boundary condition - exactly 100 elements" {
    var numbers: [100]i32 = undefined;
    for (0..100) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    for (0..100) |expected_index| {
        const pair = enumerate.next().?;
        try testing.expectEqual(@as(usize, expected_index), pair.index);
    }
}

test "enumerate: pair struct has correct fields" {
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var enumerate = Enumerate(i32, SliceIterator(i32)).init(iter);

    const pair = enumerate.next().?;
    // Check that pair has both index and value fields
    try testing.expectEqual(@as(usize, 0), pair.index);
    try testing.expectEqual(@as(i32, 42), pair.value);
}
