const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

/// Set Cover Approximation Algorithms
///
/// Set cover: given a universe U and a collection of sets S, find the smallest subcollection
/// of S that covers all elements in U.
/// Finding minimum set cover is NP-complete, but greedy algorithm gives O(log n)-approximation.

/// Computes a greedy O(log n)-approximation of minimum set cover.
/// Repeatedly picks the set that covers the most uncovered elements.
///
/// Time: O(|S| × |U|) where S = collection of sets, U = universe size
/// Space: O(|U|) for tracking covered elements
///
/// Returns: ArrayList of set indices in the cover (caller owns memory)
///
/// Algorithm:
/// 1. While there are uncovered elements:
///    - Pick the set that covers the most uncovered elements
///    - Add it to the solution
///    - Mark its elements as covered
/// 2. Approximation ratio: |C| ≤ H(n) × |OPT| where H(n) = 1 + 1/2 + ... + 1/n ≈ ln(n)
///
/// Example:
/// ```zig
/// var sets = ArrayList(ArrayList(usize)).init(allocator);
/// var s1 = ArrayList(usize).init(allocator);
/// try s1.appendSlice(&.{1, 2, 3});
/// try sets.append(s1);
/// var s2 = ArrayList(usize).init(allocator);
/// try s2.appendSlice(&.{2, 4});
/// try sets.append(s2);
///
/// var cover = try setCoverGreedy(allocator, 5, sets.items);
/// defer cover.deinit();
/// // cover contains indices of sets needed to cover {1,2,3,4}
/// ```
pub fn setCoverGreedy(
    allocator: Allocator,
    universe_size: usize,
    sets: []const []const usize,
) !ArrayList(usize) {
    var cover = ArrayList(usize).init(allocator);
    errdefer cover.deinit();

    if (sets.len == 0 or universe_size == 0) return cover;

    // Track which elements are covered
    var covered = try allocator.alloc(bool, universe_size);
    defer allocator.free(covered);
    @memset(covered, false);

    var uncovered_count = universe_size;

    // Greedy: pick set covering most uncovered elements
    while (uncovered_count > 0) {
        var max_new_coverage: usize = 0;
        var best_set_idx: ?usize = null;

        // Find set with maximum new coverage
        for (sets, 0..) |set, i| {
            var new_coverage: usize = 0;
            for (set) |elem| {
                if (elem < universe_size and !covered[elem]) {
                    new_coverage += 1;
                }
            }

            if (new_coverage > max_new_coverage) {
                max_new_coverage = new_coverage;
                best_set_idx = i;
            }
        }

        // If no set covers any new element, we're done
        if (best_set_idx == null or max_new_coverage == 0) break;

        // Add best set to cover
        const idx = best_set_idx.?;
        try cover.append(idx);

        // Mark its elements as covered
        for (sets[idx]) |elem| {
            if (elem < universe_size and !covered[elem]) {
                covered[elem] = true;
                uncovered_count -= 1;
            }
        }
    }

    return cover;
}

/// Computes set cover using a frequency-based heuristic.
/// For each uncovered element, picks the set containing it that has been used least often.
///
/// Time: O(|S| × |U|)
/// Space: O(|S|) for frequency tracking
///
/// Returns: ArrayList of set indices in the cover (caller owns memory)
///
/// Note: This is an alternative heuristic with no theoretical approximation guarantee.
/// Use setCoverGreedy for guaranteed O(log n)-approximation.
pub fn setCoverFrequency(
    allocator: Allocator,
    universe_size: usize,
    sets: []const []const usize,
) !ArrayList(usize) {
    var cover = ArrayList(usize).init(allocator);
    errdefer cover.deinit();

    if (sets.len == 0 or universe_size == 0) return cover;

    // Track which elements are covered
    var covered = try allocator.alloc(bool, universe_size);
    defer allocator.free(covered);
    @memset(covered, false);

    // Track how often each set is used
    var set_usage = try allocator.alloc(usize, sets.len);
    defer allocator.free(set_usage);
    @memset(set_usage, 0);

    var uncovered_count = universe_size;

    while (uncovered_count > 0) {
        var best_set_idx: ?usize = null;
        var min_usage: usize = std.math.maxInt(usize);

        // For first uncovered element, find set with minimum usage
        for (0..universe_size) |elem| {
            if (!covered[elem]) {
                // Find sets containing this element
                for (sets, 0..) |set, i| {
                    var contains_elem = false;
                    for (set) |e| {
                        if (e == elem) {
                            contains_elem = true;
                            break;
                        }
                    }

                    if (contains_elem and set_usage[i] < min_usage) {
                        min_usage = set_usage[i];
                        best_set_idx = i;
                    }
                }
                break; // Process one element at a time
            }
        }

        if (best_set_idx == null) break;

        const idx = best_set_idx.?;
        try cover.append(idx);
        set_usage[idx] += 1;

        // Mark elements as covered
        for (sets[idx]) |elem| {
            if (elem < universe_size and !covered[elem]) {
                covered[elem] = true;
                uncovered_count -= 1;
            }
        }
    }

    return cover;
}

/// Validates that the given set cover actually covers all elements in the universe.
///
/// Time: O(|C| × max_set_size) where C is the cover size
/// Space: O(universe_size) for tracking coverage
///
/// Returns: true if all elements are covered, false otherwise
pub fn isValidCover(
    allocator: Allocator,
    universe_size: usize,
    sets: []const []const usize,
    cover: []const usize,
) !bool {
    var covered = try allocator.alloc(bool, universe_size);
    defer allocator.free(covered);
    @memset(covered, false);

    // Mark all elements covered by selected sets
    for (cover) |set_idx| {
        if (set_idx >= sets.len) return false;
        for (sets[set_idx]) |elem| {
            if (elem < universe_size) {
                covered[elem] = true;
            }
        }
    }

    // Check all elements are covered
    for (covered) |is_covered| {
        if (!is_covered) return false;
    }

    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "set cover: empty universe" {
    const allocator = std.testing.allocator;
    const sets: []const []const usize = &.{};
    var cover = try setCoverGreedy(allocator, 0, sets);
    defer cover.deinit();
    try std.testing.expectEqual(@as(usize, 0), cover.items.len);
}

test "set cover: single set covers all" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2, 3, 4 };
    const sets = [_][]const usize{&set1};

    var cover = try setCoverGreedy(allocator, 5, &sets);
    defer cover.deinit();

    try std.testing.expectEqual(@as(usize, 1), cover.items.len);
    try std.testing.expect(try isValidCover(allocator, 5, &sets, cover.items));
}

test "set cover: disjoint sets" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{ 4, 5 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverGreedy(allocator, 6, &sets);
    defer cover.deinit();

    // Need all 3 sets to cover disjoint elements
    try std.testing.expectEqual(@as(usize, 3), cover.items.len);
    try std.testing.expect(try isValidCover(allocator, 6, &sets, cover.items));
}

test "set cover: overlapping sets" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 1, 2, 3 };
    const set3 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverGreedy(allocator, 5, &sets);
    defer cover.deinit();

    // Greedy should pick set1 (covers 3 elements), then set3 (covers 2 new)
    try std.testing.expect(cover.items.len <= 3);
    try std.testing.expect(try isValidCover(allocator, 5, &sets, cover.items));
}

test "set cover: optimal needs fewer sets" {
    const allocator = std.testing.allocator;
    // Universe: {0,1,2,3,4}
    // Set1: {0,1} Set2: {2,3} Set3: {4} Set4: {0,2,4} Set5: {1,3}
    // OPT=2: {Set4, Set5} or {Set1, Set2, Set3}=3
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{4};
    const set4 = [_]usize{ 0, 2, 4 };
    const set5 = [_]usize{ 1, 3 };
    const sets = [_][]const usize{ &set1, &set2, &set3, &set4, &set5 };

    var cover = try setCoverGreedy(allocator, 5, &sets);
    defer cover.deinit();

    // Greedy picks set4 (3 elements), then set5 (2 new) = 2 sets (optimal!)
    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(try isValidCover(allocator, 5, &sets, cover.items));
}

test "set cover: large universe" {
    const allocator = std.testing.allocator;

    // Create sets: each set covers 10 consecutive elements
    var sets_storage = ArrayList(ArrayList(usize)).init(allocator);
    defer {
        for (sets_storage.items) |*s| s.deinit();
        sets_storage.deinit();
    }

    for (0..10) |i| {
        var set = ArrayList(usize).init(allocator);
        for (0..10) |j| {
            try set.append(i * 10 + j);
        }
        try sets_storage.append(set);
    }

    var sets_slices = ArrayList([]const usize).init(allocator);
    defer sets_slices.deinit();
    for (sets_storage.items) |s| {
        try sets_slices.append(s.items);
    }

    var cover = try setCoverGreedy(allocator, 100, sets_slices.items);
    defer cover.deinit();

    // Need all 10 sets for complete coverage
    try std.testing.expectEqual(@as(usize, 10), cover.items.len);
    try std.testing.expect(try isValidCover(allocator, 100, sets_slices.items, cover.items));
}

test "set cover: redundant sets" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2, 3, 4 }; // Covers all
    const set2 = [_]usize{ 0, 1 }; // Redundant
    const set3 = [_]usize{ 2, 3 }; // Redundant
    const set4 = [_]usize{ 4 }; // Redundant
    const sets = [_][]const usize{ &set1, &set2, &set3, &set4 };

    var cover = try setCoverGreedy(allocator, 5, &sets);
    defer cover.deinit();

    // Should pick only set1
    try std.testing.expectEqual(@as(usize, 1), cover.items.len);
    try std.testing.expectEqual(@as(usize, 0), cover.items[0]);
}

test "set cover: frequency heuristic" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{ 1, 2 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverFrequency(allocator, 4, &sets);
    defer cover.deinit();

    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(try isValidCover(allocator, 4, &sets, cover.items));
}

test "set cover: greedy vs frequency comparison" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 1, 2, 3 };
    const set3 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover_greedy = try setCoverGreedy(allocator, 5, &sets);
    defer cover_greedy.deinit();

    var cover_freq = try setCoverFrequency(allocator, 5, &sets);
    defer cover_freq.deinit();

    // Both should be valid covers
    try std.testing.expect(try isValidCover(allocator, 5, &sets, cover_greedy.items));
    try std.testing.expect(try isValidCover(allocator, 5, &sets, cover_freq.items));
}

test "set cover: isValidCover detects invalid cover" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const sets = [_][]const usize{ &set1, &set2 };

    const invalid_cover = [_]usize{0}; // Missing set2
    try std.testing.expect(!try isValidCover(allocator, 4, &sets, &invalid_cover));
}

test "set cover: isValidCover accepts valid cover" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2 };

    const valid_cover = [_]usize{ 0, 1 };
    try std.testing.expect(try isValidCover(allocator, 5, &sets, &valid_cover));
}

test "set cover: sets with duplicate elements" {
    const allocator = std.testing.allocator;
    const set1 = [_]usize{ 0, 0, 1, 1 }; // Duplicates within set
    const set2 = [_]usize{ 1, 2, 2 };
    const sets = [_][]const usize{ &set1, &set2 };

    var cover = try setCoverGreedy(allocator, 3, &sets);
    defer cover.deinit();

    try std.testing.expect(cover.items.len >= 1);
    try std.testing.expect(try isValidCover(allocator, 3, &sets, cover.items));
}
