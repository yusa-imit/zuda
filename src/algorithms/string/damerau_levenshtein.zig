const std = @import("std");
const Allocator = std.mem.Allocator;

/// Damerau-Levenshtein Distance
///
/// Measures the minimum number of single-character edits (insertions, deletions,
/// substitutions, and transpositions) required to transform one string into another.
///
/// Extension of Levenshtein distance that includes transpositions (swapping two
/// adjacent characters) as a single edit operation. This makes it more suitable
/// for detecting and measuring typos, as transpositions are common typing errors.
///
/// Algorithm: Optimal String Alignment Distance (OSA) variant - O(m*n) time and space
///
/// Use cases:
/// - Spell checking and correction (typos often involve transpositions)
/// - Fuzzy string matching in search engines
/// - DNA sequence analysis (mutation detection)
/// - OCR error detection and correction
/// - Keyboard input correction (autocorrect systems)
/// - Record linkage and deduplication
///
/// Examples:
/// - "kitten" → "sitten": distance 1 (substitute k→s)
/// - "kitten" → "kittne": distance 1 (transpose n↔e)
/// - "CA" → "ABC": distance 3 (OSA variant, not full Damerau-Levenshtein)
///
/// Reference: Damerau (1964) "A technique for computer detection and correction of spelling errors"
///            Lowrance & Wagner (1975) "An Extension of the String-to-String Correction Problem"

/// Compute Damerau-Levenshtein distance (OSA variant)
/// Allows insertions, deletions, substitutions, and transpositions
/// Time: O(m*n) | Space: O(m*n)
pub fn distance(allocator: Allocator, a: []const u8, b: []const u8) !usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;

    const m = a.len;
    const n = b.len;

    // Allocate DP table (m+1) x (n+1)
    const dp = try allocator.alloc([]usize, m + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..m + 1) |i| {
        dp[i] = try allocator.alloc(usize, n + 1);
    }

    // Initialize base cases
    for (0..m + 1) |i| {
        dp[i][0] = i; // Delete all characters from a
    }
    for (0..n + 1) |j| {
        dp[0][j] = j; // Insert all characters from b
    }

    // Fill DP table
    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            const cost: usize = if (a[i - 1] == b[j - 1]) 0 else 1;

            // Standard operations: insert, delete, substitute
            dp[i][j] = @min(
                @min(
                    dp[i - 1][j] + 1, // Delete from a
                    dp[i][j - 1] + 1, // Insert into a
                ),
                dp[i - 1][j - 1] + cost, // Substitute
            );

            // Transposition: swap adjacent characters
            // Check if current and previous characters are swapped
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]) {
                dp[i][j] = @min(dp[i][j], dp[i - 2][j - 2] + 1);
            }
        }
    }

    return dp[m][n];
}

/// Compute normalized Damerau-Levenshtein distance in range [0, 1]
/// Returns 0.0 for identical strings, 1.0 for completely different strings
/// Time: O(m*n) | Space: O(m*n)
pub fn normalizedDistance(allocator: Allocator, a: []const u8, b: []const u8) !f64 {
    if (a.len == 0 and b.len == 0) return 0.0;
    const max_len = @max(a.len, b.len);
    const dist = try distance(allocator, a, b);
    return @as(f64, @floatFromInt(dist)) / @as(f64, @floatFromInt(max_len));
}

/// Compute Damerau-Levenshtein similarity in range [0, 1]
/// Returns 1.0 for identical strings, 0.0 for completely different strings
/// Time: O(m*n) | Space: O(m*n)
pub fn similarity(allocator: Allocator, a: []const u8, b: []const u8) !f64 {
    return 1.0 - try normalizedDistance(allocator, a, b);
}

/// Check if two strings are similar within a threshold distance
/// Time: O(m*n) | Space: O(m*n)
pub fn isSimilar(allocator: Allocator, a: []const u8, b: []const u8, threshold: usize) !bool {
    return try distance(allocator, a, b) <= threshold;
}

/// Find the most similar string from a list of candidates
/// Returns index of the best match, or null if no candidates
/// Time: O(k*m*n) where k = candidates.len | Space: O(m*n)
pub fn findMostSimilar(allocator: Allocator, target: []const u8, candidates: []const []const u8) !?usize {
    if (candidates.len == 0) return null;

    var min_dist: usize = std.math.maxInt(usize);
    var best_idx: usize = 0;

    for (candidates, 0..) |candidate, i| {
        const dist = try distance(allocator, target, candidate);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

/// Find all strings from candidates within a threshold distance
/// Caller owns returned ArrayList
/// Time: O(k*m*n) where k = candidates.len | Space: O(k + m*n)
pub fn findAllSimilar(allocator: Allocator, target: []const u8, candidates: []const []const u8, threshold: usize) !std.ArrayList(usize) {
    var result = std.ArrayList(usize).init(allocator);
    errdefer result.deinit();

    for (candidates, 0..) |candidate, i| {
        const dist = try distance(allocator, target, candidate);
        if (dist <= threshold) {
            try result.append(i);
        }
    }

    return result;
}

// Tests

test "damerau_levenshtein: identical strings" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(0, try distance(allocator, "hello", "hello"));
    try std.testing.expectEqual(0, try distance(allocator, "", ""));
    try std.testing.expectEqual(0, try distance(allocator, "a", "a"));
}

test "damerau_levenshtein: empty strings" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(5, try distance(allocator, "hello", ""));
    try std.testing.expectEqual(5, try distance(allocator, "", "hello"));
    try std.testing.expectEqual(0, try distance(allocator, "", ""));
}

test "damerau_levenshtein: single substitution" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(1, try distance(allocator, "kitten", "sitten"));
    try std.testing.expectEqual(1, try distance(allocator, "sitting", "sicting"));
}

test "damerau_levenshtein: single insertion" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(1, try distance(allocator, "cat", "cats"));
    try std.testing.expectEqual(1, try distance(allocator, "abc", "abcd"));
}

test "damerau_levenshtein: single deletion" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(1, try distance(allocator, "cats", "cat"));
    try std.testing.expectEqual(1, try distance(allocator, "abcd", "abc"));
}

test "damerau_levenshtein: single transposition" {
    const allocator = std.testing.allocator;
    // Adjacent character swap - key feature of Damerau-Levenshtein
    try std.testing.expectEqual(1, try distance(allocator, "ab", "ba"));
    try std.testing.expectEqual(1, try distance(allocator, "kitten", "kittne"));
    try std.testing.expectEqual(1, try distance(allocator, "acb", "abc"));
}

test "damerau_levenshtein: multiple operations" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(3, try distance(allocator, "kitten", "sitting"));
    try std.testing.expectEqual(2, try distance(allocator, "saturday", "sunday"));
    try std.testing.expectEqual(3, try distance(allocator, "rosettacode", "raisethysword"));
}

test "damerau_levenshtein: transposition vs levenshtein" {
    const allocator = std.testing.allocator;
    // "teh" → "the" is 1 transposition, would be 2 in Levenshtein (del+ins)
    try std.testing.expectEqual(1, try distance(allocator, "teh", "the"));
    // "acb" → "abc" is 1 transposition
    try std.testing.expectEqual(1, try distance(allocator, "acb", "abc"));
    // "CA" → "ABC" requires 3 operations in OSA variant
    // This is a known limitation of OSA vs full Damerau-Levenshtein
    try std.testing.expectEqual(3, try distance(allocator, "CA", "ABC"));
}

test "damerau_levenshtein: normalized distance" {
    const allocator = std.testing.allocator;
    const epsilon = 0.0001;

    // Identical strings
    try std.testing.expectApproxEqAbs(0.0, try normalizedDistance(allocator, "hello", "hello"), epsilon);

    // Completely different (5 chars)
    try std.testing.expectApproxEqAbs(1.0, try normalizedDistance(allocator, "hello", "world"), epsilon);

    // Partial similarity
    const dist = try normalizedDistance(allocator, "kitten", "sitting");
    try std.testing.expect(dist > 0.0 and dist < 1.0);

    // Empty strings
    try std.testing.expectApproxEqAbs(0.0, try normalizedDistance(allocator, "", ""), epsilon);
}

test "damerau_levenshtein: similarity" {
    const allocator = std.testing.allocator;
    const epsilon = 0.0001;

    // Identical strings
    try std.testing.expectApproxEqAbs(1.0, try similarity(allocator, "hello", "hello"), epsilon);

    // Single transposition (high similarity)
    const sim = try similarity(allocator, "teh", "the");
    try std.testing.expect(sim > 0.6 and sim <= 1.0);

    // Empty strings
    try std.testing.expectApproxEqAbs(1.0, try similarity(allocator, "", ""), epsilon);
}

test "damerau_levenshtein: is similar threshold" {
    const allocator = std.testing.allocator;

    // Within threshold
    try std.testing.expect(try isSimilar(allocator, "hello", "helo", 1));
    try std.testing.expect(try isSimilar(allocator, "teh", "the", 1));

    // Outside threshold
    try std.testing.expect(!try isSimilar(allocator, "hello", "world", 2));
    try std.testing.expect(!try isSimilar(allocator, "kitten", "sitting", 2));

    // Exact match
    try std.testing.expect(try isSimilar(allocator, "test", "test", 0));
}

test "damerau_levenshtein: find most similar" {
    const allocator = std.testing.allocator;

    const candidates = [_][]const u8{ "hello", "helo", "help", "world" };

    // "helo" is closest to "hello" (1 transposition)
    const idx1 = try findMostSimilar(allocator, "hello", &candidates);
    try std.testing.expectEqual(0, idx1.?); // "hello" exact match

    // "teh" closest to "help"?
    const candidates2 = [_][]const u8{ "help", "the", "them" };
    const idx2 = try findMostSimilar(allocator, "teh", &candidates2);
    try std.testing.expectEqual(1, idx2.?); // "the" is closest (1 transposition)

    // Empty candidates
    try std.testing.expectEqual(null, try findMostSimilar(allocator, "test", &[_][]const u8{}));
}

test "damerau_levenshtein: find all similar" {
    const allocator = std.testing.allocator;

    const candidates = [_][]const u8{ "hello", "helo", "help", "world", "helo" };

    var result = try findAllSimilar(allocator, "hello", &candidates, 1);
    defer result.deinit();

    // "hello" (exact), "helo" (del), "help" (sub), "helo" (del)
    try std.testing.expectEqual(4, result.items.len);
    try std.testing.expect(std.mem.indexOfScalar(usize, result.items, 0) != null); // hello
    try std.testing.expect(std.mem.indexOfScalar(usize, result.items, 1) != null); // helo
    try std.testing.expect(std.mem.indexOfScalar(usize, result.items, 2) != null); // help
    try std.testing.expect(std.mem.indexOfScalar(usize, result.items, 4) != null); // helo
}

test "damerau_levenshtein: spell checking use case" {
    const allocator = std.testing.allocator;

    // Simulating autocorrect for common typos
    const typos = [_]struct { input: []const u8, expected: []const u8, max_dist: usize }{
        .{ .input = "teh", .expected = "the", .max_dist = 1 }, // transposition
        .{ .input = "recieve", .expected = "receive", .max_dist = 1 }, // transposition
        .{ .input = "thier", .expected = "their", .max_dist = 1 }, // transposition
        .{ .input = "occured", .expected = "occurred", .max_dist = 1 }, // insertion needed
    };

    for (typos) |typo| {
        const dist = try distance(allocator, typo.input, typo.expected);
        try std.testing.expect(dist <= typo.max_dist);
    }
}

test "damerau_levenshtein: DNA sequence analysis" {
    const allocator = std.testing.allocator;

    // DNA sequences with mutations
    const seq1 = "ACGTACGT";
    const seq2 = "ACGTAGCT"; // substitution G→C
    const seq3 = "ACGATCGT"; // transposition TA→AT

    try std.testing.expectEqual(1, try distance(allocator, seq1, seq2));
    try std.testing.expectEqual(1, try distance(allocator, seq1, seq3));

    // More complex mutations
    const seq4 = "ACGTACGTACGT";
    const seq5 = "ACGTAGTACGT"; // deletion C
    try std.testing.expectEqual(1, try distance(allocator, seq4, seq5));
}

test "damerau_levenshtein: case sensitivity" {
    const allocator = std.testing.allocator;

    // Case sensitive by default
    try std.testing.expectEqual(1, try distance(allocator, "Hello", "hello"));
    try std.testing.expectEqual(5, try distance(allocator, "HELLO", "hello"));
}

test "damerau_levenshtein: single character strings" {
    const allocator = std.testing.allocator;

    try std.testing.expectEqual(0, try distance(allocator, "a", "a"));
    try std.testing.expectEqual(1, try distance(allocator, "a", "b"));
    try std.testing.expectEqual(1, try distance(allocator, "a", ""));
    try std.testing.expectEqual(1, try distance(allocator, "", "a"));
}

test "damerau_levenshtein: long strings" {
    const allocator = std.testing.allocator;

    const long1 = "supercalifragilisticexpialidocious";
    const long2 = "supercalifragilisitcexpialidocious"; // transposition ti↔it

    const dist = try distance(allocator, long1, long2);
    try std.testing.expectEqual(1, dist);
}

test "damerau_levenshtein: memory safety" {
    const allocator = std.testing.allocator;

    // Run multiple iterations to detect memory leaks
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        _ = try distance(allocator, "memory", "test");
        _ = try normalizedDistance(allocator, "safety", "check");
        _ = try similarity(allocator, "allocator", "tracking");

        const candidates = [_][]const u8{ "one", "two", "three" };
        _ = try findMostSimilar(allocator, "test", &candidates);

        var result = try findAllSimilar(allocator, "test", &candidates, 3);
        result.deinit();
    }

    // If we get here without leaks, memory is properly managed
    try std.testing.expect(true);
}
