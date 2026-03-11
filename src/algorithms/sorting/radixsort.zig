const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// RadixSort - Non-comparative integer sorting algorithm.
///
/// RadixSort sorts integers by processing digits from least significant (LSD) or most
/// significant (MSD) digit. It uses counting sort as a subroutine to sort on each digit.
/// Unlike comparison-based sorts, RadixSort can achieve O(n) time complexity for fixed-width
/// integer keys.
///
/// Key features:
/// - Non-comparative: processes digit values directly
/// - Stable: preserves relative order of equal elements
/// - Linear time for fixed-width integers: O(d * (n + k)) where d = digits, k = radix
/// - Requires auxiliary space: O(n + k)
///
/// Time Complexity: O(d * (n + k)) where d = number of digits, k = radix (typically 256)
/// Space Complexity: O(n + k) for count array and auxiliary buffer
///
/// Generic parameters:
/// - T: Element type (must be integer)
pub fn RadixSort(comptime T: type) type {
    return struct {
        const Self = @This();

        // Validate type at compile time
        comptime {
            switch (@typeInfo(T)) {
                .int => {},
                .comptime_int => {},
                else => @compileError("RadixSort requires integer type"),
            }
        }

        const RADIX: usize = 256; // Use byte (8-bit) radix for efficiency

        /// Sort a slice of integers using LSD (Least Significant Digit) RadixSort.
        ///
        /// Time: O(d * (n + 256)) where d = sizeof(T) | Space: O(n + 256)
        pub fn sortLSD(allocator: Allocator, items: []T) !void {
            if (items.len < 2) return;

            const num_bytes = @sizeOf(T);

            // Allocate auxiliary buffer
            const aux = try allocator.alloc(T, items.len);
            defer allocator.free(aux);

            // Process each byte from least to most significant
            for (0..num_bytes) |byte_idx| {
                try countingSortByByte(items, aux, byte_idx);
            }
        }

        /// Sort a slice of integers using MSD (Most Significant Digit) RadixSort.
        ///
        /// Time: O(d * (n + 256)) worst, O(n) best | Space: O(n + 256 + log n) stack
        pub fn sortMSD(allocator: Allocator, items: []T) !void {
            if (items.len < 2) return;

            const aux = try allocator.alloc(T, items.len);
            defer allocator.free(aux);

            const num_bytes = @sizeOf(T);
            try sortMSDRecursive(allocator, items, aux, 0, items.len, num_bytes - 1);
        }

        /// Recursive MSD sort on a specific byte position.
        fn sortMSDRecursive(
            allocator: Allocator,
            items: []T,
            aux: []T,
            start: usize,
            end: usize,
            byte_idx: usize,
        ) !void {
            if (end - start <= 1) return;

            // For small subarrays, use insertion sort
            if (end - start < 16) {
                insertionSort(items, start, end);
                return;
            }

            // Count frequencies
            var count: [RADIX + 1]usize = undefined;
            @memset(&count, 0);

            for (start..end) |i| {
                const byte = extractByte(items[i], byte_idx);
                count[@as(usize, byte) + 1] += 1;
            }

            // Cumulative counts
            for (1..RADIX + 1) |i| {
                count[i] += count[i - 1];
            }

            // Distribute
            for (start..end) |i| {
                const byte = extractByte(items[i], byte_idx);
                aux[count[@as(usize, byte)]] = items[i];
                count[@as(usize, byte)] += 1;
            }

            // Copy back
            @memcpy(items[start..end], aux[0..end - start]);

            // Recursively sort each bucket
            if (byte_idx > 0) {
                for (0..RADIX) |r| {
                    const bucket_start = if (r == 0) 0 else count[r - 1];
                    const bucket_end = count[r];
                    if (bucket_end - bucket_start > 1) {
                        try sortMSDRecursive(allocator, items, aux, start + bucket_start, start + bucket_end, byte_idx - 1);
                    }
                }
            }
        }

        /// Counting sort by a specific byte position (for LSD variant).
        fn countingSortByByte(items: []T, aux: []T, byte_idx: usize) !void {
            var count: [RADIX + 1]usize = undefined;
            @memset(&count, 0);

            // Count frequencies
            for (items) |item| {
                const byte = extractByte(item, byte_idx);
                count[@as(usize, byte) + 1] += 1;
            }

            // Cumulative counts
            for (1..RADIX + 1) |i| {
                count[i] += count[i - 1];
            }

            // Distribute to auxiliary array
            for (items) |item| {
                const byte = extractByte(item, byte_idx);
                aux[count[@as(usize, byte)]] = item;
                count[@as(usize, byte)] += 1;
            }

            // Copy back from auxiliary
            @memcpy(items, aux);
        }

        /// Extract a specific byte from an integer (handles sign for signed types).
        fn extractByte(value: T, byte_idx: usize) u8 {
            const info = @typeInfo(T);

            // For signed types, flip sign bit to handle negative numbers correctly
            if (info == .int and info.int.signedness == .signed) {
                const bit_size = @bitSizeOf(T);
                const unsigned = @as(std.meta.Int(.unsigned, bit_size), @bitCast(value));
                // Flip sign bit for proper ordering
                const adjusted = unsigned ^ (1 << (bit_size - 1));
                const shift = @as(std.math.Log2Int(std.meta.Int(.unsigned, bit_size)), @intCast(byte_idx * 8));
                return @truncate(adjusted >> shift);
            } else {
                const shift = @as(std.math.Log2Int(T), @intCast(byte_idx * 8));
                return @truncate(value >> shift);
            }
        }

        /// Insertion sort for small subarrays in MSD variant.
        fn insertionSort(items: []T, start: usize, end: usize) void {
            var i = start + 1;
            while (i < end) : (i += 1) {
                const key = items[i];
                var j = i;

                while (j > start and items[j - 1] > key) : (j -= 1) {
                    items[j] = items[j - 1];
                }

                items[j] = key;
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "RadixSort LSD - empty array" {
    var items: [0]u32 = undefined;
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);
}

test "RadixSort LSD - single element" {
    var items = [_]u32{42};
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);
    try testing.expectEqual(@as(u32, 42), items[0]);
}

test "RadixSort LSD - already sorted" {
    var items = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort LSD - reverse sorted" {
    var items = [_]u32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort LSD - random data" {
    var items = [_]u32{ 170, 45, 75, 90, 802, 24, 2, 66 };
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);

    try testing.expectEqualSlices(u32, &[_]u32{ 2, 24, 45, 66, 75, 90, 170, 802 }, &items);
}

test "RadixSort LSD - duplicates" {
    var items = [_]u32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort LSD - large values" {
    var items = [_]u32{ 4294967295, 0, 2147483647, 1, 4294967294 };
    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(testing.allocator, &items);

    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2147483647, 4294967294, 4294967295 }, &items);
}

test "RadixSort LSD - signed integers" {
    var items = [_]i32{ -5, 3, -10, 0, 8, -3, 5, -8, 1, -1 };
    const Sorter = RadixSort(i32);
    try Sorter.sortLSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }

    try testing.expectEqual(@as(i32, -10), items[0]);
    try testing.expectEqual(@as(i32, 8), items[items.len - 1]);
}

test "RadixSort LSD - u8 type" {
    var items = [_]u8{ 255, 0, 128, 64, 32, 16, 8, 4, 2, 1 };
    const Sorter = RadixSort(u8);
    try Sorter.sortLSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort LSD - u64 type" {
    var items = [_]u64{ 1000000000000, 1, 500000000000, 999999999999, 2 };
    const Sorter = RadixSort(u64);
    try Sorter.sortLSD(testing.allocator, &items);

    try testing.expectEqual(@as(u64, 1), items[0]);
    try testing.expectEqual(@as(u64, 2), items[1]);
    try testing.expectEqual(@as(u64, 500000000000), items[2]);
}

test "RadixSort MSD - empty array" {
    var items: [0]u32 = undefined;
    const Sorter = RadixSort(u32);
    try Sorter.sortMSD(testing.allocator, &items);
}

test "RadixSort MSD - single element" {
    var items = [_]u32{42};
    const Sorter = RadixSort(u32);
    try Sorter.sortMSD(testing.allocator, &items);
    try testing.expectEqual(@as(u32, 42), items[0]);
}

test "RadixSort MSD - random data" {
    var items = [_]u32{ 170, 45, 75, 90, 802, 24, 2, 66 };
    const Sorter = RadixSort(u32);
    try Sorter.sortMSD(testing.allocator, &items);

    try testing.expectEqualSlices(u32, &[_]u32{ 2, 24, 45, 66, 75, 90, 170, 802 }, &items);
}

test "RadixSort MSD - signed integers" {
    var items = [_]i32{ -5, 3, -10, 0, 8, -3, 5, -8, 1, -1 };
    const Sorter = RadixSort(i32);
    try Sorter.sortMSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }

    try testing.expectEqual(@as(i32, -10), items[0]);
    try testing.expectEqual(@as(i32, 8), items[items.len - 1]);
}

test "RadixSort MSD - duplicates" {
    var items = [_]u32{ 5, 2, 8, 2, 9, 1, 5, 5, 3, 2 };
    const Sorter = RadixSort(u32);
    try Sorter.sortMSD(testing.allocator, &items);

    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort - stress test LSD" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 10000;
    var items = try allocator.alloc(u32, n);
    defer allocator.free(items);

    // Fill with random data
    for (0..n) |i| {
        items[i] = random.int(u32);
    }

    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(allocator, items);

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort - stress test MSD" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 10000;
    var items = try allocator.alloc(u32, n);
    defer allocator.free(items);

    // Fill with random data
    for (0..n) |i| {
        items[i] = random.int(u32);
    }

    const Sorter = RadixSort(u32);
    try Sorter.sortMSD(allocator, items);

    // Verify sorted
    for (0..items.len - 1) |i| {
        try testing.expect(items[i] <= items[i + 1]);
    }
}

test "RadixSort - compare LSD vs MSD results" {
    const allocator = testing.allocator;
    var items1 = [_]u32{ 170, 45, 75, 90, 802, 24, 2, 66, 15, 33 };
    var items2: [items1.len]u32 = undefined;
    @memcpy(&items2, &items1);

    const Sorter = RadixSort(u32);
    try Sorter.sortLSD(allocator, &items1);
    try Sorter.sortMSD(allocator, &items2);

    // Both should produce same result
    try testing.expectEqualSlices(u32, &items1, &items2);
}
