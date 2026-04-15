const std = @import("std");
const testing = std.testing;

/// Jaro-Winkler Distance — String similarity metric for record linkage and fuzzy matching
///
/// The Jaro-Winkler distance is a string similarity metric measuring the edit distance
/// between two sequences. It's particularly useful for short strings such as person names.
/// The algorithm gives more favorable ratings to strings with common prefixes.
///
/// Algorithm:
/// 1. Compute Jaro distance (based on matching characters and transpositions)
/// 2. Apply Winkler prefix bonus (boosts score for common prefix up to 4 chars)
/// 3. Return similarity score in [0, 1] where 1 = identical
///
/// Properties:
/// - Symmetric: jaro(s1, s2) == jaro(s2, s1)
/// - Range: [0, 1] where 0 = no similarity, 1 = identical
/// - Prefix-sensitive: rewards common prefixes (names, addresses)
/// - Case-sensitive by default (use toLower for case-insensitive)
///
/// Use cases:
/// - Record linkage (duplicate detection in databases)
/// - Name matching (census, genealogy, customer databases)
/// - Spell checking and fuzzy search
/// - Data deduplication and entity resolution
/// - Address matching and geocoding
///
/// References:
/// - Jaro, M. (1989) "Advances in Record-Linkage Methodology"
/// - Winkler, W. (1990) "String Comparator Metrics and Enhanced Decision Rules"
/// - US Census Bureau usage for matching person names
///
/// Example:
/// ```zig
/// const similarity = jaroWinkler("MARTHA", "MARHTA");
/// // Returns ~0.961 (very similar, common prefix "MAR", one transposition)
/// ```

/// Compute Jaro similarity between two strings
///
/// Time: O(n*m) where n, m are string lengths
/// Space: O(n + m) for tracking matched characters
pub fn jaro(s1: []const u8, s2: []const u8, allocator: std.mem.Allocator) !f64 {
    if (s1.len == 0 and s2.len == 0) return 1.0;
    if (s1.len == 0 or s2.len == 0) return 0.0;

    // Match window: max(len(s1), len(s2)) / 2 - 1
    const match_distance = @max(s1.len, s2.len) / 2;
    const match_window = if (match_distance > 0) match_distance - 1 else 0;

    // Track which characters have been matched
    var s1_matches = try allocator.alloc(bool, s1.len);
    defer allocator.free(s1_matches);
    var s2_matches = try allocator.alloc(bool, s2.len);
    defer allocator.free(s2_matches);

    @memset(s1_matches, false);
    @memset(s2_matches, false);

    // Count matches
    var matches: f64 = 0;
    for (s1, 0..) |c1, i| {
        const start = if (i > match_window) i - match_window else 0;
        const end = @min(i + match_window + 1, s2.len);

        for (start..end) |j| {
            if (s2_matches[j] or c1 != s2[j]) continue;
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if (matches == 0) return 0.0;

    // Count transpositions (matched chars in different order)
    var transpositions: f64 = 0;
    var k: usize = 0;
    for (s1, 0..) |_, i| {
        if (!s1_matches[i]) continue;
        while (!s2_matches[k]) k += 1;
        if (s1[i] != s2[k]) transpositions += 1;
        k += 1;
    }

    // Jaro formula: (m/|s1| + m/|s2| + (m-t/2)/m) / 3
    const jaro_similarity = (matches / @as(f64, @floatFromInt(s1.len)) +
        matches / @as(f64, @floatFromInt(s2.len)) +
        (matches - transpositions / 2.0) / matches) / 3.0;

    return jaro_similarity;
}

/// Compute Jaro-Winkler similarity between two strings
///
/// Applies prefix bonus to Jaro distance for common prefixes (up to 4 chars)
///
/// Time: O(n*m) where n, m are string lengths
/// Space: O(n + m) for tracking matched characters
pub fn jaroWinkler(s1: []const u8, s2: []const u8, allocator: std.mem.Allocator) !f64 {
    const jaro_sim = try jaro(s1, s2, allocator);

    // Find common prefix length (up to 4 chars)
    var prefix_len: usize = 0;
    const max_prefix = @min(@min(s1.len, s2.len), 4);
    for (0..max_prefix) |i| {
        if (s1[i] == s2[i]) {
            prefix_len += 1;
        } else {
            break;
        }
    }

    // Winkler formula: jaro + (prefix_len * scaling * (1 - jaro))
    // scaling factor p = 0.1 (standard)
    const scaling: f64 = 0.1;
    return jaro_sim + @as(f64, @floatFromInt(prefix_len)) * scaling * (1.0 - jaro_sim);
}

/// Compute Jaro-Winkler similarity with custom scaling factor
///
/// Time: O(n*m) where n, m are string lengths
/// Space: O(n + m) for tracking matched characters
pub fn jaroWinklerScaled(s1: []const u8, s2: []const u8, scaling: f64, allocator: std.mem.Allocator) !f64 {
    const jaro_sim = try jaro(s1, s2, allocator);

    var prefix_len: usize = 0;
    const max_prefix = @min(@min(s1.len, s2.len), 4);
    for (0..max_prefix) |i| {
        if (s1[i] == s2[i]) {
            prefix_len += 1;
        } else {
            break;
        }
    }

    return jaro_sim + @as(f64, @floatFromInt(prefix_len)) * scaling * (1.0 - jaro_sim);
}

/// Check if two strings are similar above a threshold
///
/// Time: O(n*m) where n, m are string lengths
/// Space: O(n + m) for tracking matched characters
pub fn isSimilar(s1: []const u8, s2: []const u8, threshold: f64, allocator: std.mem.Allocator) !bool {
    const similarity = try jaroWinkler(s1, s2, allocator);
    return similarity >= threshold;
}

/// Find the most similar string from a list
///
/// Returns the index of the most similar string, or null if list is empty
///
/// Time: O(k * n * m) where k = list length, n = query length, m = avg string length
/// Space: O(n + m) for each comparison
pub fn findMostSimilar(query: []const u8, candidates: []const []const u8, allocator: std.mem.Allocator) !?usize {
    if (candidates.len == 0) return null;

    var best_index: usize = 0;
    var best_score: f64 = try jaroWinkler(query, candidates[0], allocator);

    for (candidates[1..], 1..) |candidate, i| {
        const score = try jaroWinkler(query, candidate, allocator);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }

    return best_index;
}

/// Find all strings above a similarity threshold
///
/// Returns indices of all strings with similarity >= threshold
///
/// Time: O(k * n * m) where k = list length, n = query length, m = avg string length
/// Space: O(k) for result list + O(n + m) for each comparison
pub fn findAllSimilar(query: []const u8, candidates: []const []const u8, threshold: f64, allocator: std.mem.Allocator) !std.ArrayList(usize) {
    var results = std.ArrayList(usize).init(allocator);
    errdefer results.deinit();

    for (candidates, 0..) |candidate, i| {
        const similarity = try jaroWinkler(query, candidate, allocator);
        if (similarity >= threshold) {
            try results.append(i);
        }
    }

    return results;
}

/// Compute similarity matrix for all pairs in a list
///
/// Returns symmetric matrix where result[i][j] = similarity(strings[i], strings[j])
///
/// Time: O(k² * n * m) where k = list length, n, m = avg string lengths
/// Space: O(k²) for result matrix + O(n + m) for each comparison
pub fn similarityMatrix(strings: []const []const u8, allocator: std.mem.Allocator) ![][]f64 {
    const n = strings.len;
    var matrix = try allocator.alloc([]f64, n);
    errdefer {
        for (matrix) |row| allocator.free(row);
        allocator.free(matrix);
    }

    for (0..n) |i| {
        matrix[i] = try allocator.alloc(f64, n);
        for (0..n) |j| {
            if (i == j) {
                matrix[i][j] = 1.0; // identity
            } else if (i < j) {
                matrix[i][j] = try jaroWinkler(strings[i], strings[j], allocator);
            } else {
                matrix[i][j] = matrix[j][i]; // symmetric
            }
        }
    }

    return matrix;
}

/// Free similarity matrix allocated by similarityMatrix()
///
/// Time: O(k) where k = number of strings
/// Space: O(1)
pub fn freeSimilarityMatrix(matrix: [][]f64, allocator: std.mem.Allocator) void {
    for (matrix) |row| {
        allocator.free(row);
    }
    allocator.free(matrix);
}

// ============================================================================
// Tests
// ============================================================================

test "jaro: identical strings" {
    const similarity = try jaro("MARTHA", "MARTHA", testing.allocator);
    try testing.expectApproxEqAbs(1.0, similarity, 0.001);
}

test "jaro: empty strings" {
    try testing.expectApproxEqAbs(1.0, try jaro("", "", testing.allocator), 0.001);
    try testing.expectApproxEqAbs(0.0, try jaro("", "abc", testing.allocator), 0.001);
    try testing.expectApproxEqAbs(0.0, try jaro("abc", "", testing.allocator), 0.001);
}

test "jaro: classic example MARTHA/MARHTA" {
    // MARTHA vs MARHTA: matches M,A,R,H,T,A (6), transposition TH->HT (1)
    const similarity = try jaro("MARTHA", "MARHTA", testing.allocator);
    try testing.expectApproxEqAbs(0.944, similarity, 0.001);
}

test "jaro: classic example DIXON/DICKSONX" {
    const similarity = try jaro("DIXON", "DICKSONX", testing.allocator);
    try testing.expectApproxEqAbs(0.767, similarity, 0.001);
}

test "jaro: no similarity" {
    const similarity = try jaro("abc", "xyz", testing.allocator);
    try testing.expectApproxEqAbs(0.0, similarity, 0.001);
}

test "jaro: single character match" {
    const similarity = try jaro("a", "a", testing.allocator);
    try testing.expectApproxEqAbs(1.0, similarity, 0.001);
}

test "jaro: single character no match" {
    const similarity = try jaro("a", "b", testing.allocator);
    try testing.expectApproxEqAbs(0.0, similarity, 0.001);
}

test "jaroWinkler: identical strings" {
    const similarity = try jaroWinkler("MARTHA", "MARTHA", testing.allocator);
    try testing.expectApproxEqAbs(1.0, similarity, 0.001);
}

test "jaroWinkler: classic example MARTHA/MARHTA" {
    // Jaro = 0.944, prefix = MAR (3), Winkler = 0.944 + 3*0.1*(1-0.944) = 0.961
    const similarity = try jaroWinkler("MARTHA", "MARHTA", testing.allocator);
    try testing.expectApproxEqAbs(0.961, similarity, 0.001);
}

test "jaroWinkler: classic example DWAYNE/DUANE" {
    const similarity = try jaroWinkler("DWAYNE", "DUANE", testing.allocator);
    try testing.expect(similarity > 0.82); // strong similarity with common prefix D
}

test "jaroWinkler: prefix bonus" {
    // Common prefix boosts score
    const sim1 = try jaroWinkler("abcdef", "abcxyz", testing.allocator);
    const sim2 = try jaroWinkler("abcdef", "xyzabc", testing.allocator);
    try testing.expect(sim1 > sim2); // abc prefix vs no prefix
}

test "jaroWinkler: max prefix length is 4" {
    // Only first 4 chars count for prefix bonus
    const sim1 = try jaroWinkler("abcdxxx", "abcdyyy", testing.allocator);
    const sim2 = try jaroWinkler("abcdexx", "abcdeyy", testing.allocator);
    try testing.expectApproxEqAbs(sim1, sim2, 0.001); // both have 4-char prefix
}

test "jaroWinklerScaled: custom scaling factor" {
    const sim_low = try jaroWinklerScaled("MARTHA", "MARHTA", 0.05, testing.allocator);
    const sim_high = try jaroWinklerScaled("MARTHA", "MARHTA", 0.2, testing.allocator);
    try testing.expect(sim_high > sim_low); // higher scaling = more prefix weight
}

test "isSimilar: threshold check" {
    try testing.expect(try isSimilar("MARTHA", "MARHTA", 0.9, testing.allocator));
    try testing.expect(!try isSimilar("abc", "xyz", 0.5, testing.allocator));
}

test "findMostSimilar: find best match" {
    const candidates = [_][]const u8{ "MARTHA", "MARHTA", "DIXON", "DUANE" };
    const best = try findMostSimilar("MARTHA", &candidates, testing.allocator);
    try testing.expectEqual(@as(usize, 0), best.?); // exact match
}

test "findMostSimilar: fuzzy match" {
    const candidates = [_][]const u8{ "DIXON", "DUANE", "DWAYNE", "SMITH" };
    const best = try findMostSimilar("DUANE", &candidates, testing.allocator);
    try testing.expectEqual(@as(usize, 1), best.?); // exact match
}

test "findMostSimilar: empty list" {
    const candidates: []const []const u8 = &[_][]const u8{};
    const best = try findMostSimilar("query", candidates, testing.allocator);
    try testing.expectEqual(@as(?usize, null), best);
}

test "findAllSimilar: multiple matches" {
    const candidates = [_][]const u8{ "MARTHA", "MARHTA", "DIXON", "DUANE" };
    const results = try findAllSimilar("MARTHA", &candidates, 0.9, testing.allocator);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 2), results.items.len); // MARTHA, MARHTA
    try testing.expectEqual(@as(usize, 0), results.items[0]);
    try testing.expectEqual(@as(usize, 1), results.items[1]);
}

test "findAllSimilar: no matches" {
    const candidates = [_][]const u8{ "abc", "def", "ghi" };
    const results = try findAllSimilar("xyz", &candidates, 0.5, testing.allocator);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 0), results.items.len);
}

test "similarityMatrix: small matrix" {
    const strings = [_][]const u8{ "abc", "abd", "xyz" };
    const matrix = try similarityMatrix(&strings, testing.allocator);
    defer freeSimilarityMatrix(matrix, testing.allocator);

    // Diagonal should be 1.0
    try testing.expectApproxEqAbs(1.0, matrix[0][0], 0.001);
    try testing.expectApproxEqAbs(1.0, matrix[1][1], 0.001);
    try testing.expectApproxEqAbs(1.0, matrix[2][2], 0.001);

    // abc vs abd should be similar
    try testing.expect(matrix[0][1] > 0.8);
    try testing.expectApproxEqAbs(matrix[0][1], matrix[1][0], 0.001); // symmetric

    // abc vs xyz should be dissimilar
    try testing.expect(matrix[0][2] < 0.5);
    try testing.expectApproxEqAbs(matrix[0][2], matrix[2][0], 0.001); // symmetric
}

test "name matching: census use case" {
    // Common census name variations
    const name_pairs = [_][2][]const u8{
        .{ "SMITH", "SMYTH" },
        .{ "JOHNSON", "JOHNSEN" },
        .{ "WILLIAM", "WILLIAMS" },
        .{ "ROBERT", "ROBERTO" },
    };

    for (name_pairs) |pair| {
        const similarity = try jaroWinkler(pair[0], pair[1], testing.allocator);
        try testing.expect(similarity > 0.8); // strong similarity
    }
}

test "address matching: street names" {
    const sim = try jaroWinkler("MAIN ST", "MAIN STREET", testing.allocator);
    try testing.expect(sim > 0.8); // strong similarity
}

test "case sensitivity" {
    const sim1 = try jaroWinkler("MARTHA", "MARTHA", testing.allocator);
    const sim2 = try jaroWinkler("MARTHA", "martha", testing.allocator);
    try testing.expect(sim1 > sim2); // case-sensitive by default
}

test "Unicode: ASCII only" {
    // Current implementation works on bytes, not Unicode graphemes
    const sim = try jaroWinkler("café", "cafe", testing.allocator);
    try testing.expect(sim < 1.0); // é is multi-byte
}

test "memory safety: large strings" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const s1 = try arena.allocator().alloc(u8, 1000);
    const s2 = try arena.allocator().alloc(u8, 1000);
    @memset(s1, 'a');
    @memset(s2, 'b');

    _ = try jaroWinkler(s1, s2, arena.allocator());
}

test "memory safety: repeated allocations" {
    for (0..10) |_| {
        _ = try jaroWinkler("MARTHA", "MARHTA", testing.allocator);
    }
}

test "record linkage: duplicate detection" {
    const records = [_][]const u8{
        "John Smith",
        "Jon Smith",
        "Jane Doe",
        "John Smyth",
    };

    const query = "John Smith";
    const duplicates = try findAllSimilar(query, &records, 0.85, testing.allocator);
    defer duplicates.deinit();

    try testing.expect(duplicates.items.len >= 2); // at least exact + Jon/Smyth
}

test "fuzzy search: spell correction candidates" {
    const dictionary = [_][]const u8{ "color", "colour", "dolor", "value" };
    const query = "colur"; // typo

    const best = try findMostSimilar(query, &dictionary, testing.allocator);
    const best_match = dictionary[best.?];

    // Should match "color" or "colour" (both valid)
    try testing.expect(std.mem.eql(u8, best_match, "color") or std.mem.eql(u8, best_match, "colour"));
}
