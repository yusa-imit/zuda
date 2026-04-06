const std = @import("std");
const Allocator = std.mem.Allocator;

/// Longest Consecutive Sequence
///
/// Classic problem: Find the length of the longest consecutive elements sequence in an unsorted array.
///
/// Example: [100, 4, 200, 1, 3, 2] → 4 (sequence [1, 2, 3, 4])
///
/// Algorithm approaches:
/// 1. Hash set approach (optimal): O(n) time, O(n) space
///    - Insert all elements into hash set
///    - For each element that starts a sequence (no num-1 in set), iterate consecutive numbers
///    - Only count sequences from their start to avoid redundant work
///
/// 2. Union-Find approach: O(n α(n)) time, O(n) space
///    - Build disjoint set by unioning consecutive elements
///    - Track component sizes
///    - Maximum component size is the answer
///
/// 3. Sorting approach: O(n log n) time, O(1) space
///    - Sort array
///    - Iterate and track longest consecutive run
///    - Simple but slower due to sorting
///
/// Use cases:
/// - Data analysis (finding continuous ranges)
/// - Time series analysis (gap detection)
/// - Database query optimization
/// - Genomic sequence analysis
///
/// Reference: LeetCode #128

/// Find length of longest consecutive sequence (hash set approach)
///
/// Time: O(n) where n = array length
/// Space: O(n) for hash set
///
/// Example: [100, 4, 200, 1, 3, 2] → 4 (sequence [1, 2, 3, 4])
pub fn longestConsecutive(comptime T: type, allocator: Allocator, nums: []const T) !usize {
    if (nums.len == 0) return 0;

    var set = std.AutoHashMap(T, void).init(allocator);
    defer set.deinit();

    // Insert all elements into hash set
    for (nums) |num| {
        try set.put(num, {});
    }

    var max_length: usize = 0;

    // For each element, check if it starts a sequence
    for (nums) |num| {
        // Only start counting from sequence start (no num-1 exists)
        if (!set.contains(num - 1)) {
            var current = num;
            var length: usize = 1;

            // Count consecutive numbers
            while (set.contains(current + 1)) {
                current += 1;
                length += 1;
            }

            max_length = @max(max_length, length);
        }
    }

    return max_length;
}

/// Find the actual longest consecutive sequence (hash set approach)
///
/// Time: O(n) where n = array length
/// Space: O(n) for hash set + O(k) for result where k = sequence length
///
/// Returns ArrayList containing the sequence in ascending order
pub fn findLongestConsecutive(comptime T: type, allocator: Allocator, nums: []const T) !std.ArrayList(T) {
    var result = try std.ArrayList(T).initCapacity(allocator, 0);
    errdefer result.deinit(allocator);

    if (nums.len == 0) return result;

    var set = std.AutoHashMap(T, void).init(allocator);
    defer set.deinit();

    for (nums) |num| {
        try set.put(num, {});
    }

    var max_length: usize = 0;
    var sequence_start: T = 0;

    for (nums) |num| {
        if (!set.contains(num - 1)) {
            var current = num;
            var length: usize = 1;

            while (set.contains(current + 1)) {
                current += 1;
                length += 1;
            }

            if (length > max_length) {
                max_length = length;
                sequence_start = num;
            }
        }
    }

    // Build the sequence
    var i: usize = 0;
    while (i < max_length) : (i += 1) {
        try result.append(allocator, sequence_start + @as(T, @intCast(i)));
    }

    return result;
}

/// Find length of longest consecutive sequence (sorting approach)
///
/// Time: O(n log n) due to sorting
/// Space: O(1) if we ignore sorting space
///
/// Simpler but slower alternative to hash set approach
pub fn longestConsecutiveSorted(comptime T: type, allocator: Allocator, nums: []const T) !usize {
    if (nums.len == 0) return 0;

    // Create mutable copy for sorting
    var sorted = try allocator.dupe(T, nums);
    defer allocator.free(sorted);

    std.mem.sort(T, sorted, {}, comptime std.sort.asc(T));

    var max_length: usize = 1;
    var current_length: usize = 1;

    for (sorted[1..], 0..) |num, idx| {
        const prev = sorted[idx];
        if (num == prev + 1) {
            current_length += 1;
            max_length = @max(max_length, current_length);
        } else if (num == prev) {
            // Duplicate, skip
            continue;
        } else {
            current_length = 1;
        }
    }

    return max_length;
}

/// Count number of consecutive sequences of given length
///
/// Time: O(n) where n = array length
/// Space: O(n) for hash set
///
/// Example: [1, 2, 3, 100, 101, 102, 200], length=3 → 2 sequences ([1,2,3] and [100,101,102])
pub fn countConsecutiveSequences(comptime T: type, allocator: Allocator, nums: []const T, target_length: usize) !usize {
    if (nums.len == 0 or target_length == 0) return 0;

    var set = std.AutoHashMap(T, void).init(allocator);
    defer set.deinit();

    for (nums) |num| {
        try set.put(num, {});
    }

    var count: usize = 0;

    for (nums) |num| {
        if (!set.contains(num - 1)) {
            var current = num;
            var length: usize = 1;

            while (set.contains(current + 1)) {
                current += 1;
                length += 1;
            }

            if (length == target_length) {
                count += 1;
            }
        }
    }

    return count;
}

/// Find all consecutive sequences in array
///
/// Time: O(n) where n = array length
/// Space: O(n) for hash set + O(k) for results
///
/// Returns list of sequences, each represented as ArrayList
pub fn findAllConsecutiveSequences(comptime T: type, allocator: Allocator, nums: []const T) !std.ArrayList(std.ArrayList(T)) {
    var sequences = try std.ArrayList(std.ArrayList(T)).initCapacity(allocator, 0);
    errdefer {
        for (sequences.items) |*seq| {
            seq.deinit(allocator);
        }
        sequences.deinit(allocator);
    }

    if (nums.len == 0) return sequences;

    var set = std.AutoHashMap(T, void).init(allocator);
    defer set.deinit();

    for (nums) |num| {
        try set.put(num, {});
    }

    for (nums) |num| {
        if (!set.contains(num - 1)) {
            var seq = try std.ArrayList(T).initCapacity(allocator, 0);
            errdefer seq.deinit(allocator);

            var current = num;
            try seq.append(allocator, current);

            while (set.contains(current + 1)) {
                current += 1;
                try seq.append(allocator, current);
            }

            try sequences.append(allocator, seq);
        }
    }

    return sequences;
}

/// Find length of longest consecutive sequence with bounded values
///
/// Time: O(n) where n = array length
/// Space: O(1) if range is known, O(n) otherwise
///
/// When values are bounded (e.g., 0-100), we can use array instead of hash map
/// This version uses hash map for generality
pub fn longestConsecutiveBounded(comptime T: type, allocator: Allocator, nums: []const T, min_val: T, max_val: T) !usize {
    // For small ranges, could use bit array, but hash map is general
    // This is a simplified version - in practice could optimize for bounded ranges
    _ = min_val;
    _ = max_val;
    if (nums.len == 0) return 0;
    return longestConsecutive(T, allocator, nums);
}

// Tests
test "longest consecutive - basic examples" {
    const allocator = std.testing.allocator;

    // Example 1: [100, 4, 200, 1, 3, 2] → 4
    {
        const nums = [_]i32{ 100, 4, 200, 1, 3, 2 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }

    // Example 2: [0, 3, 7, 2, 5, 8, 4, 6, 0, 1] → 9
    {
        const nums = [_]i32{ 0, 3, 7, 2, 5, 8, 4, 6, 0, 1 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 9), result);
    }
}

test "longest consecutive - edge cases" {
    const allocator = std.testing.allocator;

    // Empty array
    {
        const nums = [_]i32{};
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 0), result);
    }

    // Single element
    {
        const nums = [_]i32{1};
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 1), result);
    }

    // Two consecutive elements
    {
        const nums = [_]i32{ 1, 2 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 2), result);
    }

    // Two non-consecutive elements
    {
        const nums = [_]i32{ 1, 3 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 1), result);
    }
}

test "longest consecutive - duplicates" {
    const allocator = std.testing.allocator;

    // Duplicates in sequence
    {
        const nums = [_]i32{ 1, 2, 2, 3, 4 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }

    // All duplicates
    {
        const nums = [_]i32{ 5, 5, 5, 5 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 1), result);
    }
}

test "longest consecutive - negative numbers" {
    const allocator = std.testing.allocator;

    // Mixed negative and positive
    {
        const nums = [_]i32{ -1, 0, 1, 2 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }

    // All negative
    {
        const nums = [_]i32{ -5, -3, -4, -2 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }
}

test "longest consecutive - unsorted order" {
    const allocator = std.testing.allocator;

    // Reverse sorted
    {
        const nums = [_]i32{ 5, 4, 3, 2, 1 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 5), result);
    }

    // Random order
    {
        const nums = [_]i32{ 9, 1, 4, 7, 3, 2, 8, 5, 6 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 9), result);
    }
}

test "find longest consecutive - sequence recovery" {
    const allocator = std.testing.allocator;

    // Find actual sequence
    {
        const nums = [_]i32{ 100, 4, 200, 1, 3, 2 };
        var result = try findLongestConsecutive(i32, allocator, &nums);
        defer result.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 4), result.items.len);
        try std.testing.expectEqual(@as(i32, 1), result.items[0]);
        try std.testing.expectEqual(@as(i32, 2), result.items[1]);
        try std.testing.expectEqual(@as(i32, 3), result.items[2]);
        try std.testing.expectEqual(@as(i32, 4), result.items[3]);
    }

    // Multiple sequences, return longest
    {
        const nums = [_]i32{ 1, 2, 10, 11, 12, 13 };
        var result = try findLongestConsecutive(i32, allocator, &nums);
        defer result.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 4), result.items.len);
        try std.testing.expectEqual(@as(i32, 10), result.items[0]);
        try std.testing.expectEqual(@as(i32, 13), result.items[3]);
    }
}

test "longest consecutive sorted - alternative approach" {
    const allocator = std.testing.allocator;

    // Same results as hash set approach
    {
        const nums = [_]i32{ 100, 4, 200, 1, 3, 2 };
        const result = try longestConsecutiveSorted(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }

    // With duplicates
    {
        const nums = [_]i32{ 1, 2, 2, 3, 4 };
        const result = try longestConsecutiveSorted(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 4), result);
    }
}

test "count consecutive sequences - specific length" {
    const allocator = std.testing.allocator;

    // Count sequences of length 3
    {
        const nums = [_]i32{ 1, 2, 3, 100, 101, 102, 200 };
        const result = try countConsecutiveSequences(i32, allocator, &nums, 3);
        try std.testing.expectEqual(@as(usize, 2), result);
    }

    // No sequences of length 5
    {
        const nums = [_]i32{ 1, 2, 3, 100, 101, 102, 200 };
        const result = try countConsecutiveSequences(i32, allocator, &nums, 5);
        try std.testing.expectEqual(@as(usize, 0), result);
    }

    // Count length 1 (all elements)
    {
        const nums = [_]i32{ 1, 100, 200 };
        const result = try countConsecutiveSequences(i32, allocator, &nums, 1);
        try std.testing.expectEqual(@as(usize, 3), result);
    }
}

test "find all consecutive sequences - enumerate all" {
    const allocator = std.testing.allocator;

    // Multiple sequences
    {
        const nums = [_]i32{ 1, 2, 3, 10, 11, 20 };
        var sequences = try findAllConsecutiveSequences(i32, allocator, &nums);
        defer {
            for (sequences.items) |*seq| {
                seq.deinit(allocator);
            }
            sequences.deinit(allocator);
        }

        try std.testing.expectEqual(@as(usize, 3), sequences.items.len);

        // First sequence: [1, 2, 3]
        try std.testing.expectEqual(@as(usize, 3), sequences.items[0].items.len);
        try std.testing.expectEqual(@as(i32, 1), sequences.items[0].items[0]);

        // Second sequence: [10, 11]
        try std.testing.expectEqual(@as(usize, 2), sequences.items[1].items.len);
        try std.testing.expectEqual(@as(i32, 10), sequences.items[1].items[0]);

        // Third sequence: [20]
        try std.testing.expectEqual(@as(usize, 1), sequences.items[2].items.len);
        try std.testing.expectEqual(@as(i32, 20), sequences.items[2].items[0]);
    }
}

test "longest consecutive - large array" {
    const allocator = std.testing.allocator;

    // Large array with known sequence
    var nums = try std.ArrayList(i32).initCapacity(allocator, 102);
    defer nums.deinit(allocator);

    // Insert 1..100 in random order
    var i: i32 = 1;
    while (i <= 100) : (i += 1) {
        nums.appendAssumeCapacity(i);
    }

    // Add some outliers
    nums.appendAssumeCapacity(1000);
    nums.appendAssumeCapacity(2000);

    const result = try longestConsecutive(i32, allocator, nums.items);
    try std.testing.expectEqual(@as(usize, 100), result);
}

test "longest consecutive - u64 support" {
    const allocator = std.testing.allocator;

    // Test with u64 type
    {
        const nums = [_]u64{ 1, 2, 3, 10, 11 };
        const result = try longestConsecutive(u64, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 3), result);
    }
}

test "longest consecutive - memory safety" {
    const allocator = std.testing.allocator;

    // Multiple allocations and deallocations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const nums = [_]i32{ 1, 2, 3, 100, 101, 102 };
        const result = try longestConsecutive(i32, allocator, &nums);
        try std.testing.expectEqual(@as(usize, 3), result);

        var seq = try findLongestConsecutive(i32, allocator, &nums);
        seq.deinit(allocator);

        var sequences = try findAllConsecutiveSequences(i32, allocator, &nums);
        for (sequences.items) |*s| {
            s.deinit(allocator);
        }
        sequences.deinit(allocator);
    }
}
