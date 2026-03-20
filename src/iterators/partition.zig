//! Partition Iterator Adaptor
//!
//! The Partition adaptor consumes a base iterator and splits its elements into two
//! separate iterators based on a provided predicate function. All elements where the
//! predicate returns true go into the true_iter, and all elements where it returns
//! false go into the false_iter.
//!
//! This adaptor requires buffering: the entire base iterator is consumed upfront and
//! elements are stored in two separate dynamically allocated slices.
//!
//! ## Type Parameters
//! - T: Element type
//! - BaseIter: The underlying iterator type
//!
//! ## Time Complexity
//! - init(): O(n) where n = number of elements to consume from base iterator
//! - next(): O(1) per element (from either iterator)
//!
//! ## Space Complexity
//! - O(n) — stores all elements in two dynamically allocated slices
//!
//! ## Example
//! ```zig
//! var numbers = [_]i32{1, 2, 3, 4, 5, 6};
//! var slice_iter = SliceIterator(i32).init(&numbers);
//! var partition = Partition(i32, SliceIterator(i32)).init(allocator, slice_iter, isEven);
//! defer partition.deinit();
//! while (partition.true_iter.next()) |val| {
//!     // val will be 2, 4, 6 (even numbers)
//! }
//! while (partition.false_iter.next()) |val| {
//!     // val will be 1, 3, 5 (odd numbers)
//! }
//! ```

const std = @import("std");
const testing = std.testing;

/// Partition adaptor that splits elements into two iterators based on a predicate.
/// Time: O(n) to init, O(1) per next() call
/// Space: O(n) — buffers all elements
///
/// This is a factory function that takes the concrete base iterator type
/// and returns a struct containing two iterators.
pub fn Partition(comptime T: type, comptime _: type) type {
    return struct {
        const Self = @This();

        /// Simple iterator over a slice of buffered elements
        const BufferedSliceIterator = struct {
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

        /// Allocator for managing buffered elements
        allocator: std.mem.Allocator,
        /// True iterator (elements where predicate returns true)
        true_iter: BufferedSliceIterator,
        /// False iterator (elements where predicate returns false)
        false_iter: BufferedSliceIterator,

        /// Initialize the Partition adaptor by consuming the entire base iterator
        /// and splitting elements into true and false buffers.
        /// Time: O(n) | Space: O(n)
        pub fn init(
            allocator: std.mem.Allocator,
            base_iter: anytype,
            predicate_fn: *const fn (T) bool,
        ) !Self {
            // Single pass: collect elements into ArrayLists, then convert to owned slices
            var true_list = std.ArrayList(T).init(allocator);
            errdefer true_list.deinit();
            var false_list = std.ArrayList(T).init(allocator);
            errdefer false_list.deinit();

            var iter = base_iter;
            while (iter.next()) |value| {
                if (predicate_fn(value)) {
                    try true_list.append(value);
                } else {
                    try false_list.append(value);
                }
            }

            // Convert to owned slices (transfers ownership to caller)
            const true_slice = try true_list.toOwnedSlice();
            const false_slice = try false_list.toOwnedSlice();

            return .{
                .allocator = allocator,
                .true_iter = BufferedSliceIterator.init(true_slice),
                .false_iter = BufferedSliceIterator.init(false_slice),
            };
        }

        /// Deinitialize and free all buffered elements.
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.true_iter.slice.len > 0) {
                self.allocator.free(self.true_iter.slice);
            }
            if (self.false_iter.slice.len > 0) {
                self.allocator.free(self.false_iter.slice);
            }
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

// -- Predicate functions for testing --

fn isEven(x: i32) bool {
    return @rem(x, 2) == 0;
}

fn isOdd(x: i32) bool {
    return @rem(x, 2) != 0;
}

fn isPositive(x: i32) bool {
    return x > 0;
}

fn isNegative(x: i32) bool {
    return x < 0;
}

fn isLessThan10(x: i32) bool {
    return x < 10;
}

fn isLessThan500(x: i32) bool {
    return x < 500;
}

fn alwaysTrue(x: i32) bool {
    _ = x;
    return true;
}

fn alwaysFalse(x: i32) bool {
    _ = x;
    return false;
}

fn isFloatPositive(x: f32) bool {
    return x > 0.0;
}

fn stringLongerThan3(s: []const u8) bool {
    return s.len > 3;
}

fn isBoolTrue(x: bool) bool {
    return x;
}

// -- Unit Tests (15+ comprehensive cases) --

test "partition: basic partition [1,2,3,4,5,6] by isEven" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // Verify true_iter yields 2, 4, 6
    try testing.expectEqual(2, partition.true_iter.next());
    try testing.expectEqual(4, partition.true_iter.next());
    try testing.expectEqual(6, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // Verify false_iter yields 1, 3, 5
    try testing.expectEqual(1, partition.false_iter.next());
    try testing.expectEqual(3, partition.false_iter.next());
    try testing.expectEqual(5, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: empty iterator returns both iterators empty" {
    const allocator = testing.allocator;
    var numbers: [0]i32 = undefined;
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    try testing.expectEqual(null, partition.true_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: all elements match predicate (all even)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 2, 4, 6, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // All in true_iter
    try testing.expectEqual(2, partition.true_iter.next());
    try testing.expectEqual(4, partition.true_iter.next());
    try testing.expectEqual(6, partition.true_iter.next());
    try testing.expectEqual(8, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter is empty
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: no elements match predicate (all odd)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 3, 5, 7 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // true_iter is empty
    try testing.expectEqual(null, partition.true_iter.next());

    // All in false_iter
    try testing.expectEqual(1, partition.false_iter.next());
    try testing.expectEqual(3, partition.false_iter.next());
    try testing.expectEqual(5, partition.false_iter.next());
    try testing.expectEqual(7, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: single element matches predicate" {
    const allocator = testing.allocator;
    var numbers = [_]i32{42};
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    try testing.expectEqual(42, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: single element doesn't match predicate" {
    const allocator = testing.allocator;
    var numbers = [_]i32{41};
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    try testing.expectEqual(null, partition.true_iter.next());
    try testing.expectEqual(41, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: mixed partition with duplicates [1,1,2,2,3,3] by isOdd" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 1, 2, 2, 3, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isOdd);
    defer partition.deinit();

    // true_iter: odd numbers
    try testing.expectEqual(1, partition.true_iter.next());
    try testing.expectEqual(1, partition.true_iter.next());
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: even numbers
    try testing.expectEqual(2, partition.false_iter.next());
    try testing.expectEqual(2, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: stress test 1000 elements partition by < 500" {
    const allocator = testing.allocator;
    var numbers: [1000]i32 = undefined;
    for (0..1000) |i| {
        numbers[i] = @intCast(i);
    }

    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isLessThan500);
    defer partition.deinit();

    var true_count: usize = 0;
    while (partition.true_iter.next()) |_| {
        true_count += 1;
    }
    try testing.expectEqual(500, true_count);

    var false_count: usize = 0;
    while (partition.false_iter.next()) |_| {
        false_count += 1;
    }
    try testing.expectEqual(500, false_count);
}

test "partition: type i32 partition by isPositive" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ -5, -2, 0, 3, 8, -1 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isPositive);
    defer partition.deinit();

    // true_iter: positive
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(8, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: non-positive
    try testing.expectEqual(-5, partition.false_iter.next());
    try testing.expectEqual(-2, partition.false_iter.next());
    try testing.expectEqual(0, partition.false_iter.next());
    try testing.expectEqual(-1, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: type f32 partition by isFloatPositive" {
    const allocator = testing.allocator;
    var numbers = [_]f32{ -1.5, 2.3, 0.0, 4.7, -0.5 };
    const iter = SliceIterator(f32).init(&numbers);
    var partition = try Partition(f32, SliceIterator(f32)).init(allocator, iter, isFloatPositive);
    defer partition.deinit();

    // true_iter: positive floats
    try testing.expectEqual(2.3, partition.true_iter.next());
    try testing.expectEqual(4.7, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: non-positive floats
    try testing.expectEqual(-1.5, partition.false_iter.next());
    try testing.expectEqual(0.0, partition.false_iter.next());
    try testing.expectEqual(-0.5, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: type []const u8 partition by string length > 3" {
    const allocator = testing.allocator;
    var strings = [_][]const u8{ "hi", "hello", "bye", "world", "ab" };
    const iter = SliceIterator([]const u8).init(&strings);
    var partition = try Partition([]const u8, SliceIterator([]const u8)).init(
        allocator,
        iter,
        stringLongerThan3,
    );
    defer partition.deinit();

    // true_iter: length > 3
    try testing.expectEqualStrings("hello", partition.true_iter.next().?);
    try testing.expectEqualStrings("world", partition.true_iter.next().?);
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: length <= 3
    try testing.expectEqualStrings("hi", partition.false_iter.next().?);
    try testing.expectEqualStrings("bye", partition.false_iter.next().?);
    try testing.expectEqualStrings("ab", partition.false_iter.next().?);
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: type bool partition by isBoolTrue" {
    const allocator = testing.allocator;
    var bools = [_]bool{ false, true, false, true, true };
    const iter = SliceIterator(bool).init(&bools);
    var partition = try Partition(bool, SliceIterator(bool)).init(allocator, iter, isBoolTrue);
    defer partition.deinit();

    // true_iter: true values
    try testing.expectEqual(true, partition.true_iter.next());
    try testing.expectEqual(true, partition.true_iter.next());
    try testing.expectEqual(true, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: false values
    try testing.expectEqual(false, partition.false_iter.next());
    try testing.expectEqual(false, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: predicate always true all in true_iter" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, alwaysTrue);
    defer partition.deinit();

    // All in true_iter
    try testing.expectEqual(1, partition.true_iter.next());
    try testing.expectEqual(2, partition.true_iter.next());
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter empty
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: predicate always false all in false_iter" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, alwaysFalse);
    defer partition.deinit();

    // true_iter empty
    try testing.expectEqual(null, partition.true_iter.next());

    // All in false_iter
    try testing.expectEqual(1, partition.false_iter.next());
    try testing.expectEqual(2, partition.false_iter.next());
    try testing.expectEqual(3, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: order preservation in true_iter" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 6, 1, 4, 3, 2, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // true_iter should preserve order of even numbers: 6, 4, 2
    try testing.expectEqual(6, partition.true_iter.next());
    try testing.expectEqual(4, partition.true_iter.next());
    try testing.expectEqual(2, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter should preserve order of odd numbers: 1, 3, 5
    try testing.expectEqual(1, partition.false_iter.next());
    try testing.expectEqual(3, partition.false_iter.next());
    try testing.expectEqual(5, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: order preservation in false_iter" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 6, 1, 4, 3, 2, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isOdd);
    defer partition.deinit();

    // true_iter should preserve order of odd numbers: 1, 3, 5
    try testing.expectEqual(1, partition.true_iter.next());
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(5, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter should preserve order of even numbers: 6, 4, 2
    try testing.expectEqual(6, partition.false_iter.next());
    try testing.expectEqual(4, partition.false_iter.next());
    try testing.expectEqual(2, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: memory leak detection with std.testing.allocator" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // Just verify iterators work and deinit is called
    var count: usize = 0;
    while (partition.true_iter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(2, count);
}

test "partition: state isolation between multiple partition instances" {
    const allocator = testing.allocator;
    var numbers1 = [_]i32{ 1, 2, 3, 4 };
    var numbers2 = [_]i32{ 5, 6, 7, 8 };

    const iter1 = SliceIterator(i32).init(&numbers1);
    var partition1 = try Partition(i32, SliceIterator(i32)).init(allocator, iter1, isEven);
    defer partition1.deinit();

    const iter2 = SliceIterator(i32).init(&numbers2);
    var partition2 = try Partition(i32, SliceIterator(i32)).init(allocator, iter2, isOdd);
    defer partition2.deinit();

    // Interleave reads from both partitions
    try testing.expectEqual(2, partition1.true_iter.next());
    try testing.expectEqual(5, partition2.true_iter.next());
    try testing.expectEqual(4, partition1.true_iter.next());
    try testing.expectEqual(7, partition2.true_iter.next());

    try testing.expectEqual(1, partition1.false_iter.next());
    try testing.expectEqual(6, partition2.false_iter.next());
    try testing.expectEqual(3, partition1.false_iter.next());
    try testing.expectEqual(8, partition2.false_iter.next());
}

test "partition: chaining true_iter with another adaptor (conceptual)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // Verify true_iter yields only evens: 2, 4, 6, 8
    var all_even = true;
    while (partition.true_iter.next()) |val| {
        if (@rem(val, 2) != 0) {
            all_even = false;
        }
    }
    try testing.expect(all_even);
}

test "partition: chaining false_iter with another adaptor (conceptual)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isEven);
    defer partition.deinit();

    // Verify false_iter yields only odds: 1, 3, 5, 7
    var all_odd = true;
    while (partition.false_iter.next()) |val| {
        if (@rem(val, 2) == 0) {
            all_odd = false;
        }
    }
    try testing.expect(all_odd);
}

test "partition: negative and positive split" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ -10, 5, -3, 8, 0, -1, 7 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isPositive);
    defer partition.deinit();

    // true_iter: positive
    try testing.expectEqual(5, partition.true_iter.next());
    try testing.expectEqual(8, partition.true_iter.next());
    try testing.expectEqual(7, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: non-positive
    try testing.expectEqual(-10, partition.false_iter.next());
    try testing.expectEqual(-3, partition.false_iter.next());
    try testing.expectEqual(0, partition.false_iter.next());
    try testing.expectEqual(-1, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}

test "partition: partition with threshold (< 10)" {
    const allocator = testing.allocator;
    var numbers = [_]i32{ 5, 15, 9, 20, 3, 100, 8 };
    const iter = SliceIterator(i32).init(&numbers);
    var partition = try Partition(i32, SliceIterator(i32)).init(allocator, iter, isLessThan10);
    defer partition.deinit();

    // true_iter: < 10
    try testing.expectEqual(5, partition.true_iter.next());
    try testing.expectEqual(9, partition.true_iter.next());
    try testing.expectEqual(3, partition.true_iter.next());
    try testing.expectEqual(8, partition.true_iter.next());
    try testing.expectEqual(null, partition.true_iter.next());

    // false_iter: >= 10
    try testing.expectEqual(15, partition.false_iter.next());
    try testing.expectEqual(20, partition.false_iter.next());
    try testing.expectEqual(100, partition.false_iter.next());
    try testing.expectEqual(null, partition.false_iter.next());
}
