const std = @import("std");
const testing = std.testing;
const mem = std.mem;
const Allocator = mem.Allocator;

/// Soundex phonetic encoding algorithm
///
/// Encodes words to a phonetic representation based on pronunciation.
/// Developed by Robert C. Russell and Margaret King Odell (1918).
///
/// Algorithm:
/// 1. Retain first letter (uppercase)
/// 2. Map consonants to digits: B,F,P,V→1, C,G,J,K,Q,S,X,Z→2, D,T→3, L→4, M,N→5, R→6
/// 3. Ignore: A,E,I,O,U,H,W,Y
/// 4. Remove consecutive duplicates
/// 5. Pad with zeros to length 4
///
/// Use cases:
/// - Name matching and deduplication (census data, genealogy)
/// - Phonetic search in databases (find "Smith" when searching "Smyth")
/// - Record linkage and data quality
/// - Spell checkers and fuzzy matching
///
/// Properties:
/// - Fixed-length output (4 characters)
/// - Case-insensitive
/// - Language: English (American)
/// - Groups similar-sounding consonants
///
/// Reference: US Patent 1261167 (1918), Knuth TAOCP Vol 3

/// Standard Soundex encoding
///
/// Time: O(n) where n = input length
/// Space: O(1) - fixed 4-character output
pub fn soundex(allocator: Allocator, word: []const u8) ![]u8 {
    if (word.len == 0) return error.EmptyInput;

    var result = try allocator.alloc(u8, 4);
    errdefer allocator.free(result);

    // Initialize with zeros
    @memset(result, '0');

    // First character (uppercase)
    result[0] = std.ascii.toUpper(word[0]);
    if (!std.ascii.isAlphabetic(result[0])) {
        allocator.free(result);
        return error.InvalidCharacter;
    }

    var result_idx: usize = 1;
    var prev_code: u8 = getCode(result[0]);

    // Process remaining characters
    for (word[1..]) |c| {
        if (!std.ascii.isAlphabetic(c)) continue;

        const upper = std.ascii.toUpper(c);
        const code = getCode(upper);

        // Skip if same as previous code or if it's 0 (vowel/H/W/Y)
        if (code == 0 or code == prev_code) {
            continue;
        }

        result[result_idx] = code;
        result_idx += 1;
        prev_code = code;

        if (result_idx >= 4) break;
    }

    return result;
}

/// Get Soundex code for a character (must be uppercase)
///
/// Time: O(1)
/// Space: O(1)
fn getCode(c: u8) u8 {
    return switch (c) {
        'B', 'F', 'P', 'V' => '1',
        'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z' => '2',
        'D', 'T' => '3',
        'L' => '4',
        'M', 'N' => '5',
        'R' => '6',
        else => 0, // Vowels and H, W, Y
    };
}

/// Compare two words using Soundex encoding
///
/// Time: O(n + m) where n, m are input lengths
/// Space: O(1)
pub fn soundexMatch(allocator: Allocator, word1: []const u8, word2: []const u8) !bool {
    const code1 = try soundex(allocator, word1);
    defer allocator.free(code1);
    const code2 = try soundex(allocator, word2);
    defer allocator.free(code2);

    return mem.eql(u8, code1, code2);
}

/// Encode multiple words and return their Soundex codes
///
/// Time: O(k × n) where k = number of words, n = average word length
/// Space: O(k × 4) for output codes
pub fn soundexBatch(allocator: Allocator, words: []const []const u8) ![][]u8 {
    var result = try allocator.alloc([]u8, words.len);
    errdefer {
        for (result[0..words.len]) |code| {
            if (code.len > 0) allocator.free(code);
        }
        allocator.free(result);
    }

    for (words, 0..) |word, i| {
        result[i] = try soundex(allocator, word);
    }

    return result;
}

/// Free batch results
///
/// Time: O(k) where k = number of codes
/// Space: O(1)
pub fn freeBatch(allocator: Allocator, codes: [][]u8) void {
    for (codes) |code| {
        allocator.free(code);
    }
    allocator.free(codes);
}

// ============================================================================
// Tests
// ============================================================================

test "soundex - basic encoding" {
    const alloc = testing.allocator;

    const code = try soundex(alloc, "Robert");
    defer alloc.free(code);

    try testing.expectEqualStrings("R163", code);
}

test "soundex - classic examples" {
    const alloc = testing.allocator;

    const examples = [_]struct { word: []const u8, expected: []const u8 }{
        .{ .word = "Robert", .expected = "R163" },
        .{ .word = "Rupert", .expected = "R163" },
        .{ .word = "Rubin", .expected = "R150" },
        .{ .word = "Ashcraft", .expected = "A261" },
        .{ .word = "Ashcroft", .expected = "A261" },
        .{ .word = "Tymczak", .expected = "T522" },
        .{ .word = "Pfister", .expected = "P236" },
        .{ .word = "Honeyman", .expected = "H555" },
    };

    for (examples) |ex| {
        const code = try soundex(alloc, ex.word);
        defer alloc.free(code);
        try testing.expectEqualStrings(ex.expected, code);
    }
}

test "soundex - name pairs (should match)" {
    const alloc = testing.allocator;

    const pairs = [_]struct { a: []const u8, b: []const u8 }{
        .{ .a = "Smith", .b = "Smyth" },
        .{ .a = "Johnson", .b = "Jonson" },
        .{ .a = "Williams", .b = "Wiliams" },
        .{ .a = "Lee", .b = "Leigh" },
        .{ .a = "Jackson", .b = "Jakson" },
    };

    for (pairs) |pair| {
        const match = try soundexMatch(alloc, pair.a, pair.b);
        try testing.expect(match);
    }
}

test "soundex - name pairs (should not match)" {
    const alloc = testing.allocator;

    const pairs = [_]struct { a: []const u8, b: []const u8 }{
        .{ .a = "Smith", .b = "Schmidt" },
        .{ .a = "Johnson", .b = "Jackson" },
        .{ .a = "Williams", .b = "Wilson" },
    };

    for (pairs) |pair| {
        const match = try soundexMatch(alloc, pair.a, pair.b);
        try testing.expect(!match);
    }
}

test "soundex - edge cases" {
    const alloc = testing.allocator;

    // Single letter
    {
        const code = try soundex(alloc, "A");
        defer alloc.free(code);
        try testing.expectEqualStrings("A000", code);
    }

    // Two letters
    {
        const code = try soundex(alloc, "Ab");
        defer alloc.free(code);
        try testing.expectEqualStrings("A100", code);
    }

    // All consonants
    {
        const code = try soundex(alloc, "Bcdfghjklmnpqrstvwxyz");
        defer alloc.free(code);
        try testing.expectEqualStrings("B231", code);
    }

    // All vowels (after first)
    {
        const code = try soundex(alloc, "Aeiou");
        defer alloc.free(code);
        try testing.expectEqualStrings("A000", code);
    }
}

test "soundex - consecutive duplicates" {
    const alloc = testing.allocator;

    // Letters that map to same code should be deduplicated
    {
        const code = try soundex(alloc, "Pfeiffer");
        defer alloc.free(code);
        try testing.expectEqualStrings("P160", code);
    }

    // But separated by vowel should count both
    {
        const code = try soundex(alloc, "Pepper");
        defer alloc.free(code);
        try testing.expectEqualStrings("P160", code);
    }
}

test "soundex - case insensitive" {
    const alloc = testing.allocator;

    const code1 = try soundex(alloc, "Smith");
    defer alloc.free(code1);
    const code2 = try soundex(alloc, "SMITH");
    defer alloc.free(code2);
    const code3 = try soundex(alloc, "smith");
    defer alloc.free(code3);

    try testing.expectEqualStrings(code1, code2);
    try testing.expectEqualStrings(code1, code3);
}

test "soundex - with spaces and punctuation" {
    const alloc = testing.allocator;

    // Should skip non-alphabetic characters
    const code = try soundex(alloc, "O'Brien");
    defer alloc.free(code);
    try testing.expectEqualStrings("O165", code);
}

test "soundex - empty input" {
    const alloc = testing.allocator;

    const result = soundex(alloc, "");
    try testing.expectError(error.EmptyInput, result);
}

test "soundex - invalid first character" {
    const alloc = testing.allocator;

    const result = soundex(alloc, "123");
    try testing.expectError(error.InvalidCharacter, result);
}

test "soundex - batch encoding" {
    const alloc = testing.allocator;

    const words = [_][]const u8{ "Smith", "Johnson", "Williams", "Brown" };
    const codes = try soundexBatch(alloc, &words);
    defer freeBatch(alloc, codes);

    try testing.expectEqualStrings("S530", codes[0]);
    try testing.expectEqualStrings("J525", codes[1]);
    try testing.expectEqualStrings("W452", codes[2]);
    try testing.expectEqualStrings("B650", codes[3]);
}

test "soundex - genealogy use case" {
    const alloc = testing.allocator;

    // Common surname variations in genealogy
    const surnames = [_]struct { name: []const u8, code: []const u8 }{
        .{ .name = "Peterson", .code = "P362" },
        .{ .name = "Petersen", .code = "P362" },
        .{ .name = "MacDonald", .code = "M235" },
        .{ .name = "McDonald", .code = "M235" },
        .{ .name = "O'Neill", .code = "O540" },
        .{ .name = "Oneil", .code = "O540" },
    };

    for (surnames) |entry| {
        const code = try soundex(alloc, entry.name);
        defer alloc.free(code);
        try testing.expectEqualStrings(entry.code, code);
    }
}

test "soundex - H and W behavior" {
    const alloc = testing.allocator;

    // H and W are separators - don't contribute to code but break duplicates
    {
        const code = try soundex(alloc, "Ashcraft");
        defer alloc.free(code);
        try testing.expectEqualStrings("A261", code);
    }
}

test "soundex - length verification" {
    const alloc = testing.allocator;

    const words = [_][]const u8{ "A", "Ab", "Abc", "Abcd", "Abcde", "Abcdefghijklmnop" };

    for (words) |word| {
        const code = try soundex(alloc, word);
        defer alloc.free(code);
        try testing.expectEqual(@as(usize, 4), code.len);
    }
}

test "soundex - memory safety" {
    const alloc = testing.allocator;

    // Multiple allocations and deallocations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const code = try soundex(alloc, "TestWord");
        alloc.free(code);
    }
}
