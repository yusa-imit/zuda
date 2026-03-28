const std = @import("std");
const Allocator = std.mem.Allocator;

/// Scoring function for sequence alignment
pub const ScoreFunction = struct {
    match: i32,
    mismatch: i32,
    gap: i32,

    pub fn score(self: ScoreFunction, a: u8, b: u8) i32 {
        return if (a == b) self.match else self.mismatch;
    }
};

/// Standard DNA scoring
pub const DNA_SCORE = ScoreFunction{
    .match = 1,
    .mismatch = -1,
    .gap = -2,
};

/// Standard protein scoring (simplified BLOSUM-like)
pub const PROTEIN_SCORE = ScoreFunction{
    .match = 2,
    .mismatch = -1,
    .gap = -2,
};

/// Alignment result
pub const Alignment = struct {
    seq1: []const u8,
    seq2: []const u8,
    score: i32,
    allocator: Allocator,

    pub fn deinit(self: *Alignment) void {
        self.allocator.free(self.seq1);
        self.allocator.free(self.seq2);
    }
};

/// Needleman-Wunsch global sequence alignment
///
/// Finds the optimal global alignment between two sequences using dynamic programming.
///
/// **Algorithm**: Dynamic programming with traceback
/// **Time**: O(mn) where m, n are sequence lengths
/// **Space**: O(mn) for DP table
///
/// **Use cases**:
/// - DNA/protein sequence alignment
/// - Comparing complete sequences
/// - Finding evolutionary relationships
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `seq1`: First sequence
/// - `seq2`: Second sequence
/// - `score_fn`: Scoring function (match, mismatch, gap penalties)
///
/// **Returns**: Alignment with aligned sequences and total score
pub fn needlemanWunsch(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    score_fn: ScoreFunction,
) !Alignment {
    const m = seq1.len;
    const n = seq2.len;

    // Edge cases
    if (m == 0 and n == 0) {
        return Alignment{
            .seq1 = try allocator.dupe(u8, ""),
            .seq2 = try allocator.dupe(u8, ""),
            .score = 0,
            .allocator = allocator,
        };
    }
    if (m == 0) {
        const gaps = try allocator.alloc(u8, n);
        @memset(gaps, '-');
        return Alignment{
            .seq1 = gaps,
            .seq2 = try allocator.dupe(u8, seq2),
            .score = score_fn.gap * @as(i32, @intCast(n)),
            .allocator = allocator,
        };
    }
    if (n == 0) {
        const gaps = try allocator.alloc(u8, m);
        @memset(gaps, '-');
        return Alignment{
            .seq1 = try allocator.dupe(u8, seq1),
            .seq2 = gaps,
            .score = score_fn.gap * @as(i32, @intCast(m)),
            .allocator = allocator,
        };
    }

    // DP table: dp[i][j] = best score for seq1[0..i] and seq2[0..j]
    const dp = try allocator.alloc([]i32, m + 1);
    defer allocator.free(dp);
    for (dp) |*row| {
        row.* = try allocator.alloc(i32, n + 1);
    }
    defer for (dp) |row| allocator.free(row);

    // Initialize first row and column (gaps)
    dp[0][0] = 0;
    for (1..m + 1) |i| {
        dp[i][0] = @as(i32, @intCast(i)) * score_fn.gap;
    }
    for (1..n + 1) |j| {
        dp[0][j] = @as(i32, @intCast(j)) * score_fn.gap;
    }

    // Fill DP table
    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            const match_score = dp[i - 1][j - 1] + score_fn.score(seq1[i - 1], seq2[j - 1]);
            const delete_score = dp[i - 1][j] + score_fn.gap;
            const insert_score = dp[i][j - 1] + score_fn.gap;

            dp[i][j] = @max(@max(match_score, delete_score), insert_score);
        }
    }

    // Traceback to construct alignment
    var align1 = std.ArrayList(u8).init(allocator);
    errdefer align1.deinit();
    var align2 = std.ArrayList(u8).init(allocator);
    errdefer align2.deinit();

    var i: usize = m;
    var j: usize = n;
    while (i > 0 or j > 0) {
        if (i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + score_fn.score(seq1[i - 1], seq2[j - 1])) {
            // Match or mismatch
            try align1.append(seq1[i - 1]);
            try align2.append(seq2[j - 1]);
            i -= 1;
            j -= 1;
        } else if (i > 0 and dp[i][j] == dp[i - 1][j] + score_fn.gap) {
            // Delete from seq1 (gap in seq2)
            try align1.append(seq1[i - 1]);
            try align2.append('-');
            i -= 1;
        } else {
            // Insert into seq1 (gap in seq1)
            try align1.append('-');
            try align2.append(seq2[j - 1]);
            j -= 1;
        }
    }

    // Reverse alignments (we built them backwards)
    std.mem.reverse(u8, align1.items);
    std.mem.reverse(u8, align2.items);

    return Alignment{
        .seq1 = try align1.toOwnedSlice(),
        .seq2 = try align2.toOwnedSlice(),
        .score = dp[m][n],
        .allocator = allocator,
    };
}

/// Smith-Waterman local sequence alignment
///
/// Finds the optimal local alignment (best-matching subsequence) between two sequences.
///
/// **Algorithm**: Dynamic programming with traceback, allowing zero score
/// **Time**: O(mn) where m, n are sequence lengths
/// **Space**: O(mn) for DP table
///
/// **Use cases**:
/// - Finding conserved regions in sequences
/// - Database searching (e.g., BLAST-like)
/// - Identifying functional domains
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `seq1`: First sequence
/// - `seq2`: Second sequence
/// - `score_fn`: Scoring function (match, mismatch, gap penalties)
///
/// **Returns**: Alignment with best-matching subsequences and score
pub fn smithWaterman(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    score_fn: ScoreFunction,
) !Alignment {
    const m = seq1.len;
    const n = seq2.len;

    // Edge cases
    if (m == 0 or n == 0) {
        return Alignment{
            .seq1 = try allocator.dupe(u8, ""),
            .seq2 = try allocator.dupe(u8, ""),
            .score = 0,
            .allocator = allocator,
        };
    }

    // DP table: dp[i][j] = best local score ending at seq1[i], seq2[j]
    const dp = try allocator.alloc([]i32, m + 1);
    defer allocator.free(dp);
    for (dp) |*row| {
        row.* = try allocator.alloc(i32, n + 1);
    }
    defer for (dp) |row| allocator.free(row);

    // Initialize to zero (key difference from Needleman-Wunsch)
    for (dp) |row| {
        @memset(row, 0);
    }

    // Fill DP table, track maximum score position
    var max_score: i32 = 0;
    var max_i: usize = 0;
    var max_j: usize = 0;

    for (1..m + 1) |i| {
        for (1..n + 1) |j| {
            const match_score = dp[i - 1][j - 1] + score_fn.score(seq1[i - 1], seq2[j - 1]);
            const delete_score = dp[i - 1][j] + score_fn.gap;
            const insert_score = dp[i][j - 1] + score_fn.gap;

            // Local alignment allows starting fresh (score 0)
            dp[i][j] = @max(@max(@max(match_score, delete_score), insert_score), 0);

            if (dp[i][j] > max_score) {
                max_score = dp[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // No positive score found
    if (max_score == 0) {
        return Alignment{
            .seq1 = try allocator.dupe(u8, ""),
            .seq2 = try allocator.dupe(u8, ""),
            .score = 0,
            .allocator = allocator,
        };
    }

    // Traceback from max score position until reaching 0
    var align1 = std.ArrayList(u8).init(allocator);
    errdefer align1.deinit();
    var align2 = std.ArrayList(u8).init(allocator);
    errdefer align2.deinit();

    var i = max_i;
    var j = max_j;
    while (i > 0 and j > 0 and dp[i][j] > 0) {
        if (dp[i][j] == dp[i - 1][j - 1] + score_fn.score(seq1[i - 1], seq2[j - 1])) {
            try align1.append(seq1[i - 1]);
            try align2.append(seq2[j - 1]);
            i -= 1;
            j -= 1;
        } else if (dp[i][j] == dp[i - 1][j] + score_fn.gap) {
            try align1.append(seq1[i - 1]);
            try align2.append('-');
            i -= 1;
        } else {
            try align1.append('-');
            try align2.append(seq2[j - 1]);
            j -= 1;
        }
    }

    std.mem.reverse(u8, align1.items);
    std.mem.reverse(u8, align2.items);

    return Alignment{
        .seq1 = try align1.toOwnedSlice(),
        .seq2 = try align2.toOwnedSlice(),
        .score = max_score,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "needleman-wunsch: identical sequences" {
    const allocator = std.testing.allocator;
    const seq1 = "ACGT";
    const seq2 = "ACGT";

    var result = try needlemanWunsch(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    try std.testing.expectEqual(@as(i32, 4), result.score); // 4 matches
    try std.testing.expectEqualStrings("ACGT", result.seq1);
    try std.testing.expectEqualStrings("ACGT", result.seq2);
}

test "needleman-wunsch: completely different" {
    const allocator = std.testing.allocator;
    const seq1 = "AAA";
    const seq2 = "TTT";

    var result = try needlemanWunsch(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    try std.testing.expect(result.score < 0); // 3 mismatches
    try std.testing.expectEqual(@as(usize, 3), result.seq1.len);
    try std.testing.expectEqual(@as(usize, 3), result.seq2.len);
}

test "needleman-wunsch: with gaps" {
    const allocator = std.testing.allocator;
    const seq1 = "ACGT";
    const seq2 = "AGT";

    var result = try needlemanWunsch(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // Best alignment: AC-GT / A--GT or similar (gap in seq2)
    try std.testing.expectEqual(@as(usize, 4), result.seq1.len);
    try std.testing.expectEqual(@as(usize, 4), result.seq2.len);
}

test "needleman-wunsch: empty sequences" {
    const allocator = std.testing.allocator;

    var result1 = try needlemanWunsch(allocator, "", "", DNA_SCORE);
    defer result1.deinit();
    try std.testing.expectEqual(@as(i32, 0), result1.score);

    var result2 = try needlemanWunsch(allocator, "ACGT", "", DNA_SCORE);
    defer result2.deinit();
    try std.testing.expectEqual(@as(i32, -8), result2.score); // 4 gaps × -2

    var result3 = try needlemanWunsch(allocator, "", "ACGT", DNA_SCORE);
    defer result3.deinit();
    try std.testing.expectEqual(@as(i32, -8), result3.score);
}

test "needleman-wunsch: protein scoring" {
    const allocator = std.testing.allocator;
    const seq1 = "ACDEG";
    const seq2 = "ACDEG";

    var result = try needlemanWunsch(allocator, seq1, seq2, PROTEIN_SCORE);
    defer result.deinit();

    try std.testing.expectEqual(@as(i32, 10), result.score); // 5 matches × 2
}

test "needleman-wunsch: single character" {
    const allocator = std.testing.allocator;

    var result1 = try needlemanWunsch(allocator, "A", "A", DNA_SCORE);
    defer result1.deinit();
    try std.testing.expectEqual(@as(i32, 1), result1.score);

    var result2 = try needlemanWunsch(allocator, "A", "T", DNA_SCORE);
    defer result2.deinit();
    try std.testing.expectEqual(@as(i32, -1), result2.score);
}

test "smith-waterman: perfect local match" {
    const allocator = std.testing.allocator;
    const seq1 = "TTACGTAA";
    const seq2 = "GGACGTCC";

    var result = try smithWaterman(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // Should find "ACGT" as best local alignment
    try std.testing.expect(result.score > 0);
    try std.testing.expect(result.seq1.len >= 4);
    try std.testing.expect(result.seq2.len >= 4);
}

test "smith-waterman: no similarity" {
    const allocator = std.testing.allocator;
    const seq1 = "AAAA";
    const seq2 = "TTTT";

    var result = try smithWaterman(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // No positive local alignment (all mismatches)
    try std.testing.expectEqual(@as(i32, 0), result.score);
    try std.testing.expectEqual(@as(usize, 0), result.seq1.len);
}

test "smith-waterman: empty sequences" {
    const allocator = std.testing.allocator;

    var result1 = try smithWaterman(allocator, "", "", DNA_SCORE);
    defer result1.deinit();
    try std.testing.expectEqual(@as(i32, 0), result1.score);

    var result2 = try smithWaterman(allocator, "ACGT", "", DNA_SCORE);
    defer result2.deinit();
    try std.testing.expectEqual(@as(i32, 0), result2.score);
}

test "smith-waterman: small conserved region" {
    const allocator = std.testing.allocator;
    const seq1 = "AAAAACGTAAAA";
    const seq2 = "TTTTACGTTTTT";

    var result = try smithWaterman(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // Should find "ACGT" (4 matches = score 4)
    try std.testing.expectEqual(@as(i32, 4), result.score);
}

test "smith-waterman: protein sequences" {
    const allocator = std.testing.allocator;
    const seq1 = "LLLACDEGMMM";
    const seq2 = "NNNACDEGPPP";

    var result = try smithWaterman(allocator, seq1, seq2, PROTEIN_SCORE);
    defer result.deinit();

    // Should find "ACDEG" (5 matches × 2 = 10)
    try std.testing.expectEqual(@as(i32, 10), result.score);
}

test "smith-waterman: gap in local alignment" {
    const allocator = std.testing.allocator;
    const seq1 = "ACGTACGT";
    const seq2 = "ACGTCGT";

    var result = try smithWaterman(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // Best local may or may not include gap depending on penalty
    try std.testing.expect(result.score > 0);
}

test "needleman-wunsch: longer sequences" {
    const allocator = std.testing.allocator;
    const seq1 = "GCATGCU";
    const seq2 = "GATTACA";

    var result = try needlemanWunsch(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 7), result.seq1.len);
    try std.testing.expectEqual(@as(usize, 7), result.seq2.len);
    // Score should be reasonable (some matches, some gaps/mismatches)
    try std.testing.expect(result.score > -15 and result.score < 7);
}

test "smith-waterman: multiple possible alignments" {
    const allocator = std.testing.allocator;
    const seq1 = "ACGTACGT";
    const seq2 = "ACGTACGT";

    var result = try smithWaterman(allocator, seq1, seq2, DNA_SCORE);
    defer result.deinit();

    // Should find entire sequence
    try std.testing.expectEqual(@as(i32, 8), result.score);
    try std.testing.expectEqualStrings("ACGTACGT", result.seq1);
}
