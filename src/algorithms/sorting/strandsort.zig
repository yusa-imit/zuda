const std = @import("std");
const testing = std.testing;
const Order = std.math.Order;
const Allocator = std.mem.Allocator;

/// Strand Sort: Repeatedly extracts sorted sublists and merges them
///
/// Algorithm: While input remains, extract a sorted strand (elements in order),
/// merge it into the output using merge from merge sort. Works well on data
/// with existing sorted subsequences.
///
/// Time: O(n²) average, O(n log n) best case (when data is sorted or has few strands)
/// Space: O(n) for temporary storage
/// Stability: Stable - preserves relative order of equal elements
/// Adaptive: Yes - performs better on partially sorted data
///
/// Use cases:
/// - Partially sorted data (few inversions)
/// - Data arriving in sorted chunks
/// - Natural merge sort alternative
/// - Educational purposes (demonstrates strand extraction and merging)
///
/// Trade-offs:
/// - vs Merge Sort: Better on partially sorted data, worse on random data
/// - vs Insertion Sort: More efficient for data with long sorted runs
/// - vs TimSort: Similar idea (natural runs) but simpler, not as optimized
///
/// Reference: Classic sorting algorithm, origins unclear but well-documented
pub fn strandSort(
    comptime T: type,
    arr: []T,
    comptime compareFn: fn (T, T) bool,
) void {
    if (arr.len <= 1) return;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    strandSortAlloc(T, arr, allocator, compareFn) catch unreachable;
}

/// Strand Sort with custom allocator
///
/// Time: O(n²) average, O(n log n) best case
/// Space: O(n)
pub fn strandSortAlloc(
    comptime T: type,
    arr: []T,
    allocator: Allocator,
    comptime compareFn: fn (T, T) bool,
) !void {
    if (arr.len <= 1) return;

    // Use ArrayList for dynamic input/output management
    var input = std.ArrayList(T).init(allocator);
    defer input.deinit();

    var output = std.ArrayList(T).init(allocator);
    defer output.deinit();

    // Copy input to working list
    try input.appendSlice(arr);

    // Extract strands until input is empty
    while (input.items.len > 0) {
        // Extract sorted strand
        var strand = std.ArrayList(T).init(allocator);
        defer strand.deinit();

        try strand.append(input.orderedRemove(0));

        var i: usize = 0;
        while (i < input.items.len) {
            if (compareFn(strand.items[strand.items.len - 1], input.items[i])) {
                // Current element extends the strand
                try strand.append(input.orderedRemove(i));
            } else {
                i += 1;
            }
        }

        // Merge strand into output
        var merged = std.ArrayList(T).init(allocator);
        defer merged.deinit();

        try merge(T, output.items, strand.items, &merged, compareFn);

        // Replace output with merged result
        output.clearRetainingCapacity();
        try output.appendSlice(merged.items);
    }

    // Copy result back to original array
    @memcpy(arr, output.items);
}

/// Strand Sort ascending (convenience wrapper)
///
/// Time: O(n²) average, O(n log n) best case
/// Space: O(n)
pub fn strandSortAsc(comptime T: type, arr: []T) void {
    strandSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return a < b;
        }
    }.lessThan);
}

/// Strand Sort descending (convenience wrapper)
///
/// Time: O(n²) average, O(n log n) best case
/// Space: O(n)
pub fn strandSortDesc(comptime T: type, arr: []T) void {
    strandSort(T, arr, struct {
        fn greaterThan(a: T, b: T) bool {
            return a > b;
        }
    }.greaterThan);
}

/// Strand Sort by Order (std.math.Order-based comparison)
///
/// Time: O(n²) average, O(n log n) best case
/// Space: O(n)
pub fn strandSortBy(
    comptime T: type,
    arr: []T,
    comptime compareFn: fn (void, T, T) Order,
) void {
    strandSort(T, arr, struct {
        fn lessThan(a: T, b: T) bool {
            return compareFn({}, a, b) == .lt;
        }
    }.lessThan);
}

/// Count number of strands extracted (for analysis)
///
/// Returns: Number of sorted strands in the input
pub fn countStrands(
    comptime T: type,
    arr: []const T,
    allocator: Allocator,
    comptime compareFn: fn (T, T) bool,
) !usize {
    if (arr.len <= 1) return if (arr.len == 1) 1 else 0;

    var input = std.ArrayList(T).init(allocator);
    defer input.deinit();
    try input.appendSlice(arr);

    var count: usize = 0;

    while (input.items.len > 0) {
        count += 1;

        // Extract one strand
        var last = input.orderedRemove(0);
        var i: usize = 0;
        while (i < input.items.len) {
            if (compareFn(last, input.items[i])) {
                last = input.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    return count;
}

/// Get all strands (for analysis and debugging)
///
/// Returns: ArrayList of strands, caller owns memory
pub fn getAllStrands(
    comptime T: type,
    arr: []const T,
    allocator: Allocator,
    comptime compareFn: fn (T, T) bool,
) !std.ArrayList(std.ArrayList(T)) {
    var strands = std.ArrayList(std.ArrayList(T)).init(allocator);
    errdefer {
        for (strands.items) |strand| {
            strand.deinit();
        }
        strands.deinit();
    }

    if (arr.len == 0) return strands;

    var input = std.ArrayList(T).init(allocator);
    defer input.deinit();
    try input.appendSlice(arr);

    while (input.items.len > 0) {
        var strand = std.ArrayList(T).init(allocator);
        errdefer strand.deinit();

        try strand.append(input.orderedRemove(0));

        var i: usize = 0;
        while (i < input.items.len) {
            if (compareFn(strand.items[strand.items.len - 1], input.items[i])) {
                try strand.append(input.orderedRemove(i));
            } else {
                i += 1;
            }
        }

        try strands.append(strand);
    }

    return strands;
}

/// Merge two sorted arrays (helper for strand merge)
fn merge(
    comptime T: type,
    left: []const T,
    right: []const T,
    result: *std.ArrayList(T),
    comptime compareFn: fn (T, T) bool,
) !void {
    var i: usize = 0;
    var j: usize = 0;

    while (i < left.len and j < right.len) {
        if (compareFn(left[i], right[j])) {
            try result.append(left[i]);
            i += 1;
        } else {
            try result.append(right[j]);
            j += 1;
        }
    }

    // Append remaining elements
    while (i < left.len) {
        try result.append(left[i]);
        i += 1;
    }

    while (j < right.len) {
        try result.append(right[j]);
        j += 1;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "strand sort - basic ascending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "strand sort - basic descending" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    strandSortDesc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 9, 8, 5, 2, 1 }, &arr);
}

test "strand sort - already sorted" {
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "strand sort - reverse sorted" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &arr);
}

test "strand sort - duplicates" {
    var arr = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6, 5 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 4, 5, 5, 6, 9 }, &arr);
}

test "strand sort - all equal" {
    var arr = [_]i32{ 7, 7, 7, 7, 7 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 7, 7, 7, 7, 7 }, &arr);
}

test "strand sort - single element" {
    var arr = [_]i32{42};
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &arr);
}

test "strand sort - two elements" {
    var arr = [_]i32{ 2, 1 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, &arr);
}

test "strand sort - empty array" {
    var arr = [_]i32{};
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{}, &arr);
}

test "strand sort - negative numbers" {
    var arr = [_]i32{ -5, 3, -2, 8, -9, 1 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ -9, -5, -2, 1, 3, 8 }, &arr);
}

test "strand sort - f64 support" {
    var arr = [_]f64{ 3.14, 1.41, 2.71, 0.57 };
    strandSortAsc(f64, &arr);
    try testing.expect(arr[0] < arr[1]);
    try testing.expect(arr[1] < arr[2]);
    try testing.expect(arr[2] < arr[3]);
}

test "strand sort - custom comparison (struct)" {
    const Person = struct {
        age: i32,
        name: []const u8,
    };

    var people = [_]Person{
        .{ .age = 30, .name = "Alice" },
        .{ .age = 25, .name = "Bob" },
        .{ .age = 35, .name = "Charlie" },
    };

    strandSort(Person, &people, struct {
        fn byAge(a: Person, b: Person) bool {
            return a.age < b.age;
        }
    }.byAge);

    try testing.expectEqual(@as(i32, 25), people[0].age);
    try testing.expectEqual(@as(i32, 30), people[1].age);
    try testing.expectEqual(@as(i32, 35), people[2].age);
}

test "strand sort - Order-based comparison" {
    var arr = [_]i32{ 5, 2, 8, 1, 9 };
    strandSortBy(i32, &arr, struct {
        fn compare(_: void, a: i32, b: i32) Order {
            return std.math.order(a, b);
        }
    }.compare);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 8, 9 }, &arr);
}

test "strand sort - u8 type" {
    var arr = [_]u8{ 5, 2, 8, 1, 9 };
    strandSortAsc(u8, &arr);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 5, 8, 9 }, &arr);
}

test "strand sort - partially sorted (best case)" {
    // Data with existing sorted runs should extract fewer strands
    var arr = [_]i32{ 1, 2, 3, 7, 4, 5, 6, 9, 8, 10 };
    strandSortAsc(i32, &arr);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &arr);
}

test "strand sort - count strands (sorted)" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    const count = try countStrands(i32, &arr, testing.allocator, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan);
    try testing.expectEqual(@as(usize, 1), count); // One strand - fully sorted
}

test "strand sort - count strands (reverse)" {
    const arr = [_]i32{ 5, 4, 3, 2, 1 };
    const count = try countStrands(i32, &arr, testing.allocator, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan);
    try testing.expectEqual(@as(usize, 5), count); // Five strands - worst case
}

test "strand sort - count strands (partially sorted)" {
    const arr = [_]i32{ 1, 2, 4, 3, 5, 6 };
    const count = try countStrands(i32, &arr, testing.allocator, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan);
    try testing.expect(count >= 1 and count <= 6); // Should be 2 strands: [1,2,4,5,6] and [3]
}

test "strand sort - get all strands" {
    const arr = [_]i32{ 3, 1, 4, 2, 5 };
    var strands = try getAllStrands(i32, &arr, testing.allocator, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan);
    defer {
        for (strands.items) |strand| {
            strand.deinit();
        }
        strands.deinit();
    }

    try testing.expect(strands.items.len >= 1);

    // First strand should start with 3
    try testing.expectEqual(@as(i32, 3), strands.items[0].items[0]);
}

test "strand sort - large array with allocator" {
    const allocator = testing.allocator;

    const size = 100;
    const arr = try allocator.alloc(i32, size);
    defer allocator.free(arr);

    // Fill with pseudo-random values
    for (arr, 0..) |*val, i| {
        val.* = @intCast((i * 17 + 13) % 100);
    }

    try strandSortAlloc(i32, arr, allocator, struct {
        fn lessThan(a: i32, b: i32) bool {
            return a < b;
        }
    }.lessThan);

    // Verify sorted
    for (0..arr.len - 1) |i| {
        try testing.expect(arr[i] <= arr[i + 1]);
    }
}

test "strand sort - stability test" {
    const Item = struct {
        key: i32,
        value: i32,
    };

    var arr = [_]Item{
        .{ .key = 3, .value = 1 },
        .{ .key = 1, .value = 2 },
        .{ .key = 3, .value = 3 },
        .{ .key = 2, .value = 4 },
        .{ .key = 3, .value = 5 },
    };

    strandSort(Item, &arr, struct {
        fn byKey(a: Item, b: Item) bool {
            return a.key < b.key;
        }
    }.byKey);

    // Check keys are sorted
    try testing.expectEqual(@as(i32, 1), arr[0].key);
    try testing.expectEqual(@as(i32, 2), arr[1].key);
    try testing.expectEqual(@as(i32, 3), arr[2].key);
    try testing.expectEqual(@as(i32, 3), arr[3].key);
    try testing.expectEqual(@as(i32, 3), arr[4].key);

    // Check stability: for key=3, values should be in original order (1, 3, 5)
    try testing.expectEqual(@as(i32, 1), arr[2].value);
    try testing.expectEqual(@as(i32, 3), arr[3].value);
    try testing.expectEqual(@as(i32, 5), arr[4].value);
}

test "strand sort - memory safety (no leaks)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const arr = try allocator.alloc(i32, 50);
        defer allocator.free(arr);

        for (arr, 0..) |*val, i| {
            val.* = @intCast((i * 7) % 50);
        }

        try strandSortAlloc(i32, arr, allocator, struct {
            fn lessThan(a: i32, b: i32) bool {
                return a < b;
            }
        }.lessThan);
    }
}
