const std = @import("std");
const Allocator = std.mem.Allocator;

/// Smith-Waterman algorithm for local sequence alignment
/// Finds the most similar region between two sequences
///
/// Reference: Smith & Waterman (1981) "Identification of Common Molecular Subsequences"
/// Journal of Molecular Biology 147:195-197
///
/// Time: O(n*m) where n = len(seq1), m = len(seq2)
/// Space: O(n*m) for scoring matrix

/// Alignment scoring parameters
pub const ScoringMatrix = struct {
    match: i32 = 2,
    mismatch: i32 = -1,
    gap: i32 = -1,
};

/// Alignment result
pub const Alignment = struct {
    aligned_seq1: []const u8,
    aligned_seq2: []const u8,
    score: i32,
    start1: usize, // Start position in seq1
    start2: usize, // Start position in seq2
    allocator: Allocator,

    pub fn deinit(self: *Alignment) void {
        self.allocator.free(self.aligned_seq1);
        self.allocator.free(self.aligned_seq2);
    }
};

/// Compute Smith-Waterman local alignment
///
/// Time: O(n*m) where n = len(seq1), m = len(seq2)
/// Space: O(n*m) for scoring and traceback matrices
pub fn localAlign(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    scoring: ScoringMatrix,
) !Alignment {
    if (seq1.len == 0 or seq2.len == 0) {
        return error.EmptySequence;
    }

    const rows = seq1.len + 1;
    const cols = seq2.len + 1;

    // Allocate scoring matrix
    const score_matrix = try allocator.alloc([]i32, rows);
    defer {
        for (score_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(score_matrix);
    }
    for (score_matrix) |*row| {
        row.* = try allocator.alloc(i32, cols);
        @memset(row.*, 0);
    }

    // Fill scoring matrix
    var max_score: i32 = 0;
    var max_i: usize = 0;
    var max_j: usize = 0;

    for (1..rows) |i| {
        for (1..cols) |j| {
            const match_mismatch = if (seq1[i - 1] == seq2[j - 1])
                scoring.match
            else
                scoring.mismatch;

            const diagonal = score_matrix[i - 1][j - 1] + match_mismatch;
            const up = score_matrix[i - 1][j] + scoring.gap;
            const left = score_matrix[i][j - 1] + scoring.gap;

            // Smith-Waterman: allow 0 (local alignment)
            score_matrix[i][j] = @max(@max(@max(diagonal, up), left), 0);

            // Track maximum score position
            if (score_matrix[i][j] > max_score) {
                max_score = score_matrix[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback from max score to 0
    var aligned1 = try std.ArrayList(u8).initCapacity(allocator, @min(seq1.len, seq2.len));
    errdefer aligned1.deinit(allocator);
    var aligned2 = try std.ArrayList(u8).initCapacity(allocator, @min(seq1.len, seq2.len));
    errdefer aligned2.deinit(allocator);

    var i = max_i;
    var j = max_j;

    while (i > 0 and j > 0 and score_matrix[i][j] > 0) {
        const current = score_matrix[i][j];
        const diagonal = score_matrix[i - 1][j - 1];
        const up = score_matrix[i - 1][j];
        const left = score_matrix[i][j - 1];

        const match_mismatch = if (seq1[i - 1] == seq2[j - 1])
            scoring.match
        else
            scoring.mismatch;

        if (current == diagonal + match_mismatch) {
            // Match or mismatch
            try aligned1.append(allocator, seq1[i - 1]);
            try aligned2.append(allocator, seq2[j - 1]);
            i -= 1;
            j -= 1;
        } else if (current == up + scoring.gap) {
            // Gap in seq2
            try aligned1.append(allocator, seq1[i - 1]);
            try aligned2.append(allocator, '-');
            i -= 1;
        } else if (current == left + scoring.gap) {
            // Gap in seq1
            try aligned1.append(allocator, '-');
            try aligned2.append(allocator, seq2[j - 1]);
            j -= 1;
        } else {
            // Shouldn't reach here in valid alignment
            break;
        }
    }

    // Reverse the alignments (built backwards)
    std.mem.reverse(u8, aligned1.items);
    std.mem.reverse(u8, aligned2.items);

    return Alignment{
        .aligned_seq1 = try aligned1.toOwnedSlice(allocator),
        .aligned_seq2 = try aligned2.toOwnedSlice(allocator),
        .score = max_score,
        .start1 = i, // Position after last traceback step
        .start2 = j,
        .allocator = allocator,
    };
}

/// Compute similarity percentage between two sequences
///
/// Time: O(n*m)
/// Space: O(n*m)
pub fn similarity(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    scoring: ScoringMatrix,
) !f64 {
    var alignment = try localAlign(allocator, seq1, seq2, scoring);
    defer alignment.deinit();

    // Count matches in alignment
    var matches: usize = 0;
    const len = @min(alignment.aligned_seq1.len, alignment.aligned_seq2.len);
    for (0..len) |i| {
        if (alignment.aligned_seq1[i] == alignment.aligned_seq2[i] and
            alignment.aligned_seq1[i] != '-')
        {
            matches += 1;
        }
    }

    // Similarity as percentage of alignment length
    if (len == 0) return 0.0;
    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(len));
}

/// Find best local alignment score without returning alignment
///
/// Time: O(n*m)
/// Space: O(m) - uses space-optimized version
pub fn score(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    scoring: ScoringMatrix,
) !i32 {
    if (seq1.len == 0 or seq2.len == 0) {
        return error.EmptySequence;
    }

    const cols = seq2.len + 1;

    // Use two rows for space optimization
    var prev_row = try allocator.alloc(i32, cols);
    defer allocator.free(prev_row);
    var curr_row = try allocator.alloc(i32, cols);
    defer allocator.free(curr_row);

    @memset(prev_row, 0);
    @memset(curr_row, 0);

    var max_score: i32 = 0;

    for (1..seq1.len + 1) |i| {
        curr_row[0] = 0;

        for (1..cols) |j| {
            const match_mismatch = if (seq1[i - 1] == seq2[j - 1])
                scoring.match
            else
                scoring.mismatch;

            const diagonal = prev_row[j - 1] + match_mismatch;
            const up = prev_row[j] + scoring.gap;
            const left = curr_row[j - 1] + scoring.gap;

            curr_row[j] = @max(@max(@max(diagonal, up), left), 0);
            max_score = @max(max_score, curr_row[j]);
        }

        // Swap rows
        const temp = prev_row;
        prev_row = curr_row;
        curr_row = temp;
    }

    return max_score;
}

/// Check if two sequences have significant local similarity
/// (score above threshold)
///
/// Time: O(n*m)
/// Space: O(m)
pub fn hasSimilarRegion(
    allocator: Allocator,
    seq1: []const u8,
    seq2: []const u8,
    scoring: ScoringMatrix,
    threshold: i32,
) !bool {
    const s = try score(allocator, seq1, seq2, scoring);
    return s >= threshold;
}

// Tests
test "Smith-Waterman: basic alignment" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "ACACACTA",
        "AGCACACA",
        .{},
    );
    defer alignment.deinit();

    try std.testing.expect(alignment.score > 0);
    try std.testing.expect(alignment.aligned_seq1.len > 0);
    try std.testing.expect(alignment.aligned_seq2.len > 0);
}

test "Smith-Waterman: identical sequences" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "ACGT",
        "ACGT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    // Perfect match should give score = len * match_score
    try std.testing.expectEqual(@as(i32, 8), alignment.score);
    try std.testing.expectEqualStrings("ACGT", alignment.aligned_seq1);
    try std.testing.expectEqualStrings("ACGT", alignment.aligned_seq2);
}

test "Smith-Waterman: no similarity" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "AAAA",
        "TTTT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    // No good local alignment (all mismatches)
    try std.testing.expect(alignment.score == 0);
}

test "Smith-Waterman: local alignment in longer sequences" {
    const allocator = std.testing.allocator;

    // Find "ACACA" in middle
    var alignment = try localAlign(
        allocator,
        "GGGGACACAGGGG",
        "TTTTACACATTTT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    try std.testing.expect(alignment.score > 0);
    // Should find the ACACA region
    try std.testing.expect(alignment.aligned_seq1.len >= 5);
}

test "Smith-Waterman: with gaps" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "ACGT",
        "AGT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    try std.testing.expect(alignment.score > 0);
    // Alignment should have a gap
    const has_gap = std.mem.indexOfScalar(u8, alignment.aligned_seq1, '-') != null or
        std.mem.indexOfScalar(u8, alignment.aligned_seq2, '-') != null;
    try std.testing.expect(has_gap);
}

test "Smith-Waterman: similarity percentage" {
    const allocator = std.testing.allocator;

    // Identical sequences should have 100% similarity
    const sim1 = try similarity(allocator, "ACGT", "ACGT", .{});
    try std.testing.expectApproxEqRel(1.0, sim1, 0.01);

    // Partial match
    const sim2 = try similarity(allocator, "ACGT", "ACGG", .{});
    try std.testing.expect(sim2 > 0.5);

    // No similarity
    const sim3 = try similarity(allocator, "AAAA", "TTTT", .{});
    try std.testing.expectApproxEqRel(0.0, sim3, 0.01);
}

test "Smith-Waterman: score function" {
    const allocator = std.testing.allocator;

    const s1 = try score(allocator, "ACGT", "ACGT", .{ .match = 2, .mismatch = -1, .gap = -1 });
    try std.testing.expectEqual(@as(i32, 8), s1);

    const s2 = try score(allocator, "AAAA", "TTTT", .{ .match = 2, .mismatch = -1, .gap = -1 });
    try std.testing.expectEqual(@as(i32, 0), s2);
}

test "Smith-Waterman: hasSimilarRegion" {
    const allocator = std.testing.allocator;

    const similar = try hasSimilarRegion(
        allocator,
        "ACGTACGT",
        "ACGTACGT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
        10,
    );
    try std.testing.expect(similar);

    const not_similar = try hasSimilarRegion(
        allocator,
        "AAAA",
        "TTTT",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
        10,
    );
    try std.testing.expect(!not_similar);
}

test "Smith-Waterman: custom scoring matrix" {
    const allocator = std.testing.allocator;

    const scoring_high_gap = ScoringMatrix{
        .match = 5,
        .mismatch = -2,
        .gap = -5,
    };

    var alignment = try localAlign(
        allocator,
        "ACGT",
        "AGT",
        scoring_high_gap,
    );
    defer alignment.deinit();

    // High gap penalty should discourage gaps
    try std.testing.expect(alignment.score > 0);
}

test "Smith-Waterman: DNA sequences" {
    const allocator = std.testing.allocator;

    const dna1 = "ATGCATGCATGC";
    const dna2 = "ATGCATGC";

    var alignment = try localAlign(allocator, dna1, dna2, .{});
    defer alignment.deinit();

    try std.testing.expect(alignment.score > 0);
    // dna2 should be fully aligned within dna1
    try std.testing.expect(alignment.aligned_seq2.len == dna2.len);
}

test "Smith-Waterman: protein sequences" {
    const allocator = std.testing.allocator;

    const prot1 = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV";
    const prot2 = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLS";

    var alignment = try localAlign(allocator, prot1, prot2, .{});
    defer alignment.deinit();

    try std.testing.expect(alignment.score > 0);
}

test "Smith-Waterman: error on empty sequence" {
    const allocator = std.testing.allocator;

    const result = localAlign(allocator, "", "ACGT", .{});
    try std.testing.expectError(error.EmptySequence, result);

    const result2 = localAlign(allocator, "ACGT", "", .{});
    try std.testing.expectError(error.EmptySequence, result2);
}

test "Smith-Waterman: single character" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(allocator, "A", "A", .{});
    defer alignment.deinit();

    try std.testing.expectEqual(@as(i32, 2), alignment.score);
    try std.testing.expectEqualStrings("A", alignment.aligned_seq1);
    try std.testing.expectEqualStrings("A", alignment.aligned_seq2);
}

test "Smith-Waterman: memory safety" {
    const allocator = std.testing.allocator;

    for (0..10) |_| {
        var alignment = try localAlign(
            allocator,
            "ACGTACGTACGT",
            "ACGTACGT",
            .{},
        );
        alignment.deinit();
    }
}

test "Smith-Waterman: alignment start positions" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "GGGGACGTTTTT",
        "XXXXACGTXXXX",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    // Should find ACGT alignment starting around position 4
    try std.testing.expect(alignment.start1 <= 4);
    try std.testing.expect(alignment.start2 <= 4);
}

test "Smith-Waterman: long sequences" {
    const allocator = std.testing.allocator;

    var seq1 = try allocator.alloc(u8, 100);
    defer allocator.free(seq1);
    var seq2 = try allocator.alloc(u8, 100);
    defer allocator.free(seq2);

    // Create repeating pattern
    for (0..100) |i| {
        seq1[i] = "ACGT"[i % 4];
        seq2[i] = "ACGT"[i % 4];
    }

    var alignment = try localAlign(allocator, seq1, seq2, .{});
    defer alignment.deinit();

    // Should get perfect alignment
    try std.testing.expectEqual(@as(i32, 200), alignment.score);
}

test "Smith-Waterman: different alphabet" {
    const allocator = std.testing.allocator;

    // Test with numbers represented as strings
    var alignment = try localAlign(
        allocator,
        "12341234",
        "12341234",
        .{ .match = 1, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    try std.testing.expectEqual(@as(i32, 8), alignment.score);
}

test "Smith-Waterman: case sensitivity" {
    const allocator = std.testing.allocator;

    var alignment = try localAlign(
        allocator,
        "ACGT",
        "acgt",
        .{ .match = 2, .mismatch = -1, .gap = -1 },
    );
    defer alignment.deinit();

    // All mismatches (case sensitive)
    try std.testing.expectEqual(@as(i32, 0), alignment.score);
}
