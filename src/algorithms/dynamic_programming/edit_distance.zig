const std = @import("std");
const Allocator = std.mem.Allocator;

/// Edit Distance (Levenshtein Distance)
///
/// Measures the minimum number of single-character edits (insertions, deletions, substitutions)
/// required to transform one string into another.
///
/// Consumer: zr uses this for fuzzy command matching (src/util/levenshtein.zig)

/// Operation type for edit sequence reconstruction
pub const EditOp = enum {
    match, // Characters match, no cost
    substitute, // Replace character
    insert, // Insert character
    delete, // Delete character
};

pub const Edit = struct {
    op: EditOp,
    pos_a: usize, // Position in string a
    pos_b: usize, // Position in string b
    char_a: ?u8, // Character from a (null for insert)
    char_b: ?u8, // Character from b (null for delete)
};

/// Compute Levenshtein distance
/// Time: O(m*n) | Space: O(m*n)
pub fn distance(allocator: Allocator, a: []const u8, b: []const u8) !usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;
    const m = a.len;
    const n = b.len;

    // Allocate DP table
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

            dp[i][j] = @min(
                @min(
                    dp[i - 1][j] + 1, // Delete from a
                    dp[i][j - 1] + 1, // Insert into a
                ),
                dp[i - 1][j - 1] + cost, // Substitute
            );
        }
    }

    const result = dp[m][n];
    return result;
}

/// Compute edit distance with space optimization
/// Time: O(m*n) | Space: O(min(m,n))
pub fn distanceOptimized(allocator: Allocator, a: []const u8, b: []const u8) !usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;

    // Use shorter sequence for space optimization
    const shorter = if (a.len <= b.len) a else b;
    const longer = if (a.len <= b.len) b else a;
    const n = shorter.len;

    var prev = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev);
    var curr = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr);

    // Initialize
    for (0..n + 1) |j| {
        prev[j] = j;
    }

    for (longer, 1..) |long_char, i| {
        curr[0] = i;
        for (shorter, 1..) |short_char, j| {
            const cost: usize = if (long_char == short_char) 0 else 1;

            curr[j] = @min(
                @min(
                    prev[j] + 1, // Delete
                    curr[j - 1] + 1, // Insert
                ),
                prev[j - 1] + cost, // Substitute
            );
        }
        std.mem.swap([]usize, &prev, &curr);
    }

    return prev[n];
}

/// Compute edit distance and return the sequence of edits
/// Time: O(m*n) | Space: O(m*n)
pub fn distanceWithEdits(allocator: Allocator, a: []const u8, b: []const u8) !struct {
    distance: usize,
    edits: []Edit,
} {
    if (a.len == 0 and b.len == 0) {
        return .{ .distance = 0, .edits = try allocator.alloc(Edit, 0) };
    }

    const m = a.len;
    const n = b.len;

    // Build DP table
    const dp = try allocator.alloc([]usize, m + 1);
    defer {
        for (dp) |row| allocator.free(row);
        allocator.free(dp);
    }

    for (0..m + 1) |i| {
        dp[i] = try allocator.alloc(usize, n + 1);
    }

    for (0..m + 1) |i| {
        dp[i][0] = i;
    }
    for (0..n + 1) |j| {
        dp[0][j] = j;
    }

    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            const cost: usize = if (a[i - 1] == b[j - 1]) 0 else 1;

            dp[i][j] = @min(
                @min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                ),
                dp[i - 1][j - 1] + cost,
            );
        }
    }

    const dist = dp[m][n];

    // Backtrack to reconstruct edits
    var edits: std.ArrayList(Edit) = .{};
    defer edits.deinit(allocator);

    var i = m;
    var j = n;

    while (i > 0 or j > 0) {
        if (i > 0 and j > 0 and a[i - 1] == b[j - 1]) {
            // Match
            try edits.append(allocator, .{
                .op = .match,
                .pos_a = i - 1,
                .pos_b = j - 1,
                .char_a = a[i - 1],
                .char_b = b[j - 1],
            });
            i -= 1;
            j -= 1;
        } else {
            var min_cost: usize = std.math.maxInt(usize);
            var min_op: EditOp = .substitute;

            if (i > 0) {
                const del_cost = dp[i - 1][j];
                if (del_cost < min_cost) {
                    min_cost = del_cost;
                    min_op = .delete;
                }
            }

            if (j > 0) {
                const ins_cost = dp[i][j - 1];
                if (ins_cost < min_cost) {
                    min_cost = ins_cost;
                    min_op = .insert;
                }
            }

            if (i > 0 and j > 0) {
                const sub_cost = dp[i - 1][j - 1];
                if (sub_cost < min_cost) {
                    min_cost = sub_cost;
                    min_op = .substitute;
                }
            }

            switch (min_op) {
                .delete => {
                    try edits.append(allocator, .{
                        .op = .delete,
                        .pos_a = i - 1,
                        .pos_b = j,
                        .char_a = a[i - 1],
                        .char_b = null,
                    });
                    i -= 1;
                },
                .insert => {
                    try edits.append(allocator, .{
                        .op = .insert,
                        .pos_a = i,
                        .pos_b = j - 1,
                        .char_a = null,
                        .char_b = b[j - 1],
                    });
                    j -= 1;
                },
                .substitute => {
                    try edits.append(allocator, .{
                        .op = .substitute,
                        .pos_a = i - 1,
                        .pos_b = j - 1,
                        .char_a = a[i - 1],
                        .char_b = b[j - 1],
                    });
                    i -= 1;
                    j -= 1;
                },
                .match => unreachable,
            }
        }
    }

    // Reverse to get forward order
    std.mem.reverse(Edit, edits.items);

    return .{ .distance = dist, .edits = try edits.toOwnedSlice(allocator) };
}

/// Compute similarity ratio (0.0 to 1.0, where 1.0 is identical)
/// Time: O(m*n) | Space: O(min(m,n))
pub fn similarity(allocator: Allocator, a: []const u8, b: []const u8) !f64 {
    if (a.len == 0 and b.len == 0) return 1.0;

    const dist = try distanceOptimized(allocator, a, b);
    const max_len = @max(a.len, b.len);

    return 1.0 - (@as(f64, @floatFromInt(dist)) / @as(f64, @floatFromInt(max_len)));
}

/// Check if two strings are within a given edit distance threshold
/// Time: O(m*n) early termination possible | Space: O(n)
/// Returns true if distance <= threshold
pub fn withinThreshold(allocator: Allocator, a: []const u8, b: []const u8, threshold: usize) !bool {
    const dist = try distanceOptimized(allocator, a, b);
    return dist <= threshold;
}

// ============================================================================
// Tests
// ============================================================================

test "Edit distance: empty strings" {
    try std.testing.expectEqual(0, try distance(std.testing.allocator, "", ""));
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "abc", ""));
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "", "abc"));

    try std.testing.expectEqual(0, try distanceOptimized(std.testing.allocator, "", ""));
    try std.testing.expectEqual(3, try distanceOptimized(std.testing.allocator, "abc", ""));
}

test "Edit distance: identical strings" {
    const s = "hello";
    try std.testing.expectEqual(0, try distance(std.testing.allocator, s, s));
    try std.testing.expectEqual(0, try distanceOptimized(std.testing.allocator, s, s));

    const result = try distanceWithEdits(std.testing.allocator, s, s);
    defer std.testing.allocator.free(result.edits);
    try std.testing.expectEqual(0, result.distance);
}

test "Edit distance: single character difference" {
    try std.testing.expectEqual(1, try distance(std.testing.allocator, "cat", "bat")); // Substitute
    try std.testing.expectEqual(1, try distance(std.testing.allocator, "cat", "cats")); // Insert
    try std.testing.expectEqual(1, try distance(std.testing.allocator, "cats", "cat")); // Delete

    try std.testing.expectEqual(1, try distanceOptimized(std.testing.allocator, "cat", "bat"));
    try std.testing.expectEqual(1, try distanceOptimized(std.testing.allocator, "cat", "cats"));
}

test "Edit distance: classic examples" {
    // kitten -> sitting: 3 (k->s, e->i, insert g)
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "kitten", "sitting"));
    try std.testing.expectEqual(3, try distanceOptimized(std.testing.allocator, "kitten", "sitting"));

    // saturday -> sunday: 3
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "saturday", "sunday"));
    try std.testing.expectEqual(3, try distanceOptimized(std.testing.allocator, "saturday", "sunday"));

    // rosettacode -> raisethysword: 8
    try std.testing.expectEqual(8, try distance(std.testing.allocator, "rosettacode", "raisethysword"));
}

test "Edit distance: completely different strings" {
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "abc", "xyz"));
    try std.testing.expectEqual(3, try distanceOptimized(std.testing.allocator, "abc", "xyz"));
}

test "Edit distance: one is substring of other" {
    try std.testing.expectEqual(3, try distance(std.testing.allocator, "test", "testing")); // Insert 'i', 'n', 'g'
    try std.testing.expectEqual(3, try distanceOptimized(std.testing.allocator, "test", "testing"));
}

test "Edit distance: with edits reconstruction" {
    const result = try distanceWithEdits(std.testing.allocator, "kitten", "sitting");
    defer std.testing.allocator.free(result.edits);

    try std.testing.expectEqual(3, result.distance);
    try std.testing.expect(result.edits.len > 0);

    // Verify we have substitutes and inserts
    var has_substitute = false;
    var has_insert = false;
    for (result.edits) |edit| {
        if (edit.op == .substitute) has_substitute = true;
        if (edit.op == .insert) has_insert = true;
    }
    try std.testing.expect(has_substitute);
    try std.testing.expect(has_insert);
}

test "Edit distance: similarity ratio" {
    try std.testing.expectApproxEqAbs(1.0, try similarity(std.testing.allocator, "test", "test"), 0.001);
    try std.testing.expectApproxEqAbs(0.0, try similarity(std.testing.allocator, "abcd", "efgh"), 0.001);

    const sim = try similarity(std.testing.allocator, "kitten", "sitting");
    try std.testing.expect(sim > 0.5 and sim < 0.7); // ~57%
}

test "Edit distance: threshold check" {
    try std.testing.expect(try withinThreshold(std.testing.allocator, "test", "test", 0));
    try std.testing.expect(try withinThreshold(std.testing.allocator, "test", "testing", 3));
    try std.testing.expect(!try withinThreshold(std.testing.allocator, "test", "testing", 1));
}

test "Edit distance: case sensitivity" {
    try std.testing.expectEqual(1, try distance(std.testing.allocator, "Test", "test")); // Only first char differs
    try std.testing.expectEqual(1, try distance(std.testing.allocator, "Test", "Fest"));
}

test "Edit distance: longer strings" {
    const a = "The quick brown fox jumps over the lazy dog";
    const b = "The quick brown fox jumped over the lazy dog";

    try std.testing.expectEqual(2, try distance(std.testing.allocator, a, b)); // s->ed (substitute + insert)
    try std.testing.expectEqual(2, try distanceOptimized(std.testing.allocator, a, b));
}

test "Edit distance: consumer use case - fuzzy command matching (zr)" {
    // zr uses edit distance for command name fuzzy matching
    const command = "build";
    const typo1 = "biuld"; // 2 edits
    const typo2 = "buidl"; // 2 edits
    const typo3 = "buil"; // 1 edit (delete d)

    try std.testing.expectEqual(2, try distance(std.testing.allocator, command, typo1));
    try std.testing.expectEqual(2, try distance(std.testing.allocator, command, typo2));
    try std.testing.expectEqual(1, try distance(std.testing.allocator, command, typo3));

    // Should suggest "build" for all of these within threshold 2
    try std.testing.expect(try withinThreshold(std.testing.allocator, command, typo1, 2));
    try std.testing.expect(try withinThreshold(std.testing.allocator, command, typo2, 2));
    try std.testing.expect(try withinThreshold(std.testing.allocator, command, typo3, 2));
}

test "Edit distance: transpositions not optimized" {
    // Note: Standard Levenshtein doesn't optimize for transpositions
    // "ab" -> "ba" requires 2 substitutions, not 1 transposition
    // (Damerau-Levenshtein would handle this as 1 operation)
    try std.testing.expectEqual(2, try distance(std.testing.allocator, "ab", "ba"));
}

test "Edit distance: unicode aware (bytes)" {
    // Note: This operates on bytes, not Unicode codepoints
    const a = "café";
    const b = "cafe";

    // 'é' is 2 bytes in UTF-8, so distance > 1
    const dist = try distance(std.testing.allocator, a, b);
    try std.testing.expect(dist >= 1);
}

test "Edit distance: space optimization correctness" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n = 50;
    const a = try allocator.alloc(u8, n);
    defer allocator.free(a);
    const b = try allocator.alloc(u8, n);
    defer allocator.free(b);

    for (0..n) |i| {
        a[i] = 'a' + @as(u8, @intCast(random.intRangeAtMost(u8, 0, 25)));
        b[i] = 'a' + @as(u8, @intCast(random.intRangeAtMost(u8, 0, 25)));
    }

    const dist1 = try distance(std.testing.allocator, a, b);
    const dist2 = try distanceOptimized(std.testing.allocator, a, b);

    try std.testing.expectEqual(dist1, dist2);
}

test "Edit distance: performance on moderate strings" {
    const a = "AGGTACGTACGTTACGATCGATCGATCGATCGATCGATCG";
    const b = "ACGTACGTTACGATCGATCGATCGATCGATCGATCGATCG";

    const dist = try distanceOptimized(std.heap.page_allocator, a, b);
    try std.testing.expect(dist <= 5); // Should be small
}
