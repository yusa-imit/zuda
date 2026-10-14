//! Set cover approximation algorithms.
//!
//! Set cover: given a universe U and a collection of sets S, find the smallest subcollection of
//! S that covers all elements of U. The minimum is NP-complete; the greedy rule gives an
//! H(n)-approximation.
//!
//! Ownership: every returned `ArrayList(usize)` is owned by the caller, who releases it with
//! `deinit(gpa)` using the same allocator. On error nothing is leaked.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

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
/// const s1 = [_]usize{ 1, 2, 3 };
/// const s2 = [_]usize{ 2, 4 };
/// const sets = [_][]const usize{ &s1, &s2 };
///
/// var cover = try setCoverGreedy(gpa, 5, &sets);
/// defer cover.deinit(gpa);
///
/// // cover contains indices of sets needed to cover {1,2,3,4}
/// ```
pub fn setCoverGreedy(
    gpa: Allocator,
    universe_size: usize,
    sets: []const []const usize,
) !ArrayList(usize) {
    var cover: ArrayList(usize) = .empty;
    errdefer cover.deinit(gpa);

    if (sets.len == 0 or universe_size == 0) return cover;

    // Track which elements are covered.
    const covered = try gpa.alloc(bool, universe_size);
    defer gpa.free(covered);

    @memset(covered, false);

    // An element is counted once per candidate set: `stamp[elem] == round` marks "seen".
    const stamp = try gpa.alloc(u64, universe_size);
    defer gpa.free(stamp);

    @memset(stamp, 0);

    var uncovered_count = universe_size;
    var round: u64 = 0;

    // Each pass either covers at least one new element or stops, so `universe_size` bounds it.
    for (0..universe_size) |_| {
        if (uncovered_count == 0) break;

        var max_new_coverage: usize = 0;
        var best_set_idx: usize = 0;

        // Find the set with maximum new coverage; duplicates within a set count once.
        for (sets, 0..) |set, i| {
            round += 1;
            var new_coverage: usize = 0;
            for (set) |elem| {
                if (elem < universe_size and !covered[elem] and stamp[elem] != round) {
                    stamp[elem] = round;
                    new_coverage += 1;
                }
            }

            if (new_coverage > max_new_coverage) {
                max_new_coverage = new_coverage;
                best_set_idx = i;
            }
        }

        // No set covers any new element: the remaining elements are not coverable.
        if (max_new_coverage == 0) break;

        try cover.append(gpa, best_set_idx);

        // Mark its elements as covered.
        for (sets[best_set_idx]) |elem| {
            if (elem < universe_size and !covered[elem]) {
                covered[elem] = true;
                uncovered_count -= 1;
            }
        }
    }

    assert(uncovered_count <= universe_size);
    assert(cover.items.len <= sets.len);
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
    gpa: Allocator,
    universe_size: usize,
    sets: []const []const usize,
) !ArrayList(usize) {
    var cover: ArrayList(usize) = .empty;
    errdefer cover.deinit(gpa);

    if (sets.len == 0 or universe_size == 0) return cover;

    // Track which elements are covered (or skipped because no set contains them).
    const covered = try gpa.alloc(bool, universe_size);
    defer gpa.free(covered);

    @memset(covered, false);

    // Track how often each set is used.
    const set_usage = try gpa.alloc(usize, sets.len);
    defer gpa.free(set_usage);

    @memset(set_usage, 0);

    var uncovered_count = universe_size;

    // Each pass retires at least one element, so `universe_size` bounds the loop.
    for (0..universe_size) |_| {
        if (uncovered_count == 0) break;

        const elem = std.mem.findScalar(bool, covered, false).?;
        const best_set_idx = leastUsedSetContaining(sets, set_usage, elem) orelse {
            // No set contains this element: skip it so later elements can still be covered.
            covered[elem] = true;
            uncovered_count -= 1;
            continue;
        };
        try cover.append(gpa, best_set_idx);
        set_usage[best_set_idx] += 1;

        // Mark elements as covered
        for (sets[best_set_idx]) |member| {
            if (member < universe_size and !covered[member]) {
                covered[member] = true;
                uncovered_count -= 1;
            }
        }
    }

    assert(uncovered_count == 0);
    assert(cover.items.len <= universe_size);
    return cover;
}

/// Returns the index of the set containing `elem` with the lowest usage count (first wins ties),
/// or null when no set contains `elem`.
fn leastUsedSetContaining(
    sets: []const []const usize,
    set_usage: []const usize,
    elem: usize,
) ?usize {
    assert(sets.len == set_usage.len);
    assert(sets.len > 0);

    var best_set_idx: ?usize = null;
    var min_usage: usize = std.math.maxInt(usize);
    for (sets, 0..) |set, i| {
        const contains_elem = std.mem.findScalar(usize, set, elem) != null;
        if (contains_elem and set_usage[i] < min_usage) {
            min_usage = set_usage[i];
            best_set_idx = i;
        }
    }
    return best_set_idx;
}

/// Validates that the given set cover actually covers all elements in the universe.
///
/// Time: O(|C| × max_set_size) where C is the cover size
/// Space: O(universe_size) for tracking coverage
///
/// Returns: true if all elements are covered, false otherwise
pub fn isValidCover(
    gpa: Allocator,
    universe_size: usize,
    sets: []const []const usize,
    cover: []const usize,
) !bool {
    const covered = try gpa.alloc(bool, universe_size);
    defer gpa.free(covered);

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
    const gpa = std.testing.allocator;
    const sets: []const []const usize = &.{};
    var cover = try setCoverGreedy(gpa, 0, sets);
    defer cover.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), cover.items.len);
}

test "set cover: single set covers all" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2, 3, 4 };
    const sets = [_][]const usize{&set1};

    var cover = try setCoverGreedy(gpa, 5, &sets);
    defer cover.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 1), cover.items.len);
    try std.testing.expect(try isValidCover(gpa, 5, &sets, cover.items));
}

test "set cover: disjoint sets" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{ 4, 5 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverGreedy(gpa, 6, &sets);
    defer cover.deinit(gpa);

    // Need all 3 sets to cover disjoint elements
    try std.testing.expectEqual(@as(usize, 3), cover.items.len);
    try std.testing.expect(try isValidCover(gpa, 6, &sets, cover.items));
}

test "set cover: overlapping sets" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 1, 2, 3 };
    const set3 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverGreedy(gpa, 5, &sets);
    defer cover.deinit(gpa);

    // Greedy should pick set1 (covers 3 elements), then set3 (covers 2 new)
    try std.testing.expect(cover.items.len <= 3);
    try std.testing.expect(try isValidCover(gpa, 5, &sets, cover.items));
}

test "set cover: optimal needs fewer sets" {
    const gpa = std.testing.allocator;
    // Universe: {0,1,2,3,4}
    // Set1: {0,1} Set2: {2,3} Set3: {4} Set4: {0,2,4} Set5: {1,3}
    // OPT=2: {Set4, Set5} or {Set1, Set2, Set3}=3
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{4};
    const set4 = [_]usize{ 0, 2, 4 };
    const set5 = [_]usize{ 1, 3 };
    const sets = [_][]const usize{ &set1, &set2, &set3, &set4, &set5 };

    var cover = try setCoverGreedy(gpa, 5, &sets);
    defer cover.deinit(gpa);

    // Greedy picks set4 (3 elements), then set5 (2 new) = 2 sets (optimal!)
    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(try isValidCover(gpa, 5, &sets, cover.items));
}

test "set cover: large universe" {
    const gpa = std.testing.allocator;

    // Create sets: each set covers 10 consecutive elements
    var sets_storage: ArrayList(ArrayList(usize)) = .empty;
    defer {
        for (sets_storage.items) |*s| s.deinit(gpa);
        sets_storage.deinit(gpa);
    }

    for (0..10) |i| {
        var set: ArrayList(usize) = .empty;
        for (0..10) |j| {
            try set.append(gpa, i * 10 + j);
        }
        try sets_storage.append(gpa, set);
    }

    var sets_slices: ArrayList([]const usize) = .empty;
    defer sets_slices.deinit(gpa);
    for (sets_storage.items) |s| {
        try sets_slices.append(gpa, s.items);
    }

    var cover = try setCoverGreedy(gpa, 100, sets_slices.items);
    defer cover.deinit(gpa);

    // Need all 10 sets for complete coverage
    try std.testing.expectEqual(@as(usize, 10), cover.items.len);
    try std.testing.expect(try isValidCover(gpa, 100, sets_slices.items, cover.items));
}

test "set cover: redundant sets" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2, 3, 4 }; // Covers all
    const set2 = [_]usize{ 0, 1 }; // Redundant
    const set3 = [_]usize{ 2, 3 }; // Redundant
    const set4 = [_]usize{4}; // Redundant
    const sets = [_][]const usize{ &set1, &set2, &set3, &set4 };

    var cover = try setCoverGreedy(gpa, 5, &sets);
    defer cover.deinit(gpa);

    // Should pick only set1
    try std.testing.expectEqual(@as(usize, 1), cover.items.len);
    try std.testing.expectEqual(@as(usize, 0), cover.items[0]);
}

test "set cover: frequency heuristic" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const set3 = [_]usize{ 1, 2 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover = try setCoverFrequency(gpa, 4, &sets);
    defer cover.deinit(gpa);

    try std.testing.expect(cover.items.len >= 2);
    try std.testing.expect(try isValidCover(gpa, 4, &sets, cover.items));
}

test "set cover: greedy vs frequency comparison" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 1, 2, 3 };
    const set3 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2, &set3 };

    var cover_greedy = try setCoverGreedy(gpa, 5, &sets);
    defer cover_greedy.deinit(gpa);

    var cover_freq = try setCoverFrequency(gpa, 5, &sets);
    defer cover_freq.deinit(gpa);

    // Both should be valid covers
    try std.testing.expect(try isValidCover(gpa, 5, &sets, cover_greedy.items));
    try std.testing.expect(try isValidCover(gpa, 5, &sets, cover_freq.items));
}

test "set cover: isValidCover detects invalid cover" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1 };
    const set2 = [_]usize{ 2, 3 };
    const sets = [_][]const usize{ &set1, &set2 };

    const invalid_cover = [_]usize{0}; // Missing set2
    try std.testing.expect(!try isValidCover(gpa, 4, &sets, &invalid_cover));
}

test "set cover: isValidCover accepts valid cover" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 1, 2 };
    const set2 = [_]usize{ 3, 4 };
    const sets = [_][]const usize{ &set1, &set2 };

    const valid_cover = [_]usize{ 0, 1 };
    try std.testing.expect(try isValidCover(gpa, 5, &sets, &valid_cover));
}

test "set cover: sets with duplicate elements" {
    const gpa = std.testing.allocator;
    const set1 = [_]usize{ 0, 0, 1, 1 }; // Duplicates within set
    const set2 = [_]usize{ 1, 2, 2 };
    const sets = [_][]const usize{ &set1, &set2 };

    var cover = try setCoverGreedy(gpa, 3, &sets);
    defer cover.deinit(gpa);

    try std.testing.expect(cover.items.len >= 1);
    try std.testing.expect(try isValidCover(gpa, 3, &sets, cover.items));
}

test "set cover: duplicates inside a set count once" {
    const gpa = std.testing.allocator;
    // Set 0 repeats element 0 four times (one distinct element); set 1 covers all three.
    const set0 = [_]usize{ 0, 0, 0, 0 };
    const set1 = [_]usize{ 0, 1, 2 };
    const sets = [_][]const usize{ &set0, &set1 };

    var cover = try setCoverGreedy(gpa, 3, &sets);
    defer cover.deinit(gpa);

    try std.testing.expectEqualSlices(usize, &.{1}, cover.items);
}

test "set cover: frequency skips elements no set contains" {
    const gpa = std.testing.allocator;
    // Element 0 is in no set; element 1 and 2 must still be covered.
    const set0 = [_]usize{ 1, 2 };
    const sets = [_][]const usize{&set0};

    var cover = try setCoverFrequency(gpa, 3, &sets);
    defer cover.deinit(gpa);

    try std.testing.expectEqualSlices(usize, &.{0}, cover.items);
    try std.testing.expect(!try isValidCover(gpa, 3, &sets, cover.items));
}

test "set cover: no leaks when an allocation fails" {
    const Runner = struct {
        fn run(gpa: Allocator) !void {
            const set0 = [_]usize{ 0, 1 };
            const set1 = [_]usize{ 1, 2, 3 };
            const set2 = [_]usize{ 3, 4 };
            const sets = [_][]const usize{ &set0, &set1, &set2 };

            var greedy = try setCoverGreedy(gpa, 5, &sets);
            defer greedy.deinit(gpa);

            var frequency = try setCoverFrequency(gpa, 5, &sets);
            defer frequency.deinit(gpa);

            try std.testing.expect(try isValidCover(gpa, 5, &sets, greedy.items));
            try std.testing.expect(try isValidCover(gpa, 5, &sets, frequency.items));
        }
    };

    try std.testing.checkAllAllocationFailures(std.testing.allocator, Runner.run, .{});
}
