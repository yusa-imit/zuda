// Metaphone — Phonetic Encoding Algorithm
//
// Metaphone is a phonetic algorithm for indexing words by their English pronunciation.
// Developed by Lawrence Philips (1990) as an improvement over Soundex.
//
// Algorithm:
// 1. Converts words to a phonetic code based on pronunciation rules
// 2. Handles complex English pronunciation patterns (silent letters, digraphs, etc.)
// 3. Produces variable-length codes (up to max_length, typically 4)
// 4. More accurate than Soundex for English words
//
// Reference: Lawrence Philips (1990), Computer Language Magazine Vol. 7, No. 12
//
// Time: O(n) where n is the input length
// Space: O(m) where m is the maximum output length (typically 4)

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Encode a word using the Metaphone algorithm
///
/// Returns the phonetic code as a dynamically allocated string.
/// Caller owns the returned memory and must free it.
///
/// Time: O(n) where n is the input length
/// Space: O(m) where m is max_length
pub fn metaphone(allocator: Allocator, word: []const u8, max_length: usize) ![]u8 {
    if (word.len == 0) return try allocator.dupe(u8, "");

    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    var i: usize = 0;
    const len = word.len;

    // Convert to uppercase for processing
    const upper = try allocator.alloc(u8, len);
    defer allocator.free(upper);
    for (word, 0..) |c, idx| {
        upper[idx] = std.ascii.toUpper(c);
    }

    // Skip non-alphabetic prefix
    while (i < len and !std.ascii.isAlphabetic(upper[i])) : (i += 1) {}
    if (i >= len) return try allocator.dupe(u8, "");

    // Handle initial special cases
    if (len >= 2) {
        // Initial KN, GN, PN, AE, WR -> drop first letter
        const prefix = upper[i .. i + 2];
        if (std.mem.eql(u8, prefix, "KN") or
            std.mem.eql(u8, prefix, "GN") or
            std.mem.eql(u8, prefix, "PN") or
            std.mem.eql(u8, prefix, "AE") or
            std.mem.eql(u8, prefix, "WR"))
        {
            i += 1;
        }
    }

    // Initial WH -> W
    if (i + 1 < len and upper[i] == 'W' and upper[i + 1] == 'H') {
        try result.append('W');
        i += 2;
    }
    // Initial X -> S
    else if (i < len and upper[i] == 'X') {
        try result.append('S');
        i += 1;
    }

    // Process remaining characters
    while (i < len and result.items.len < max_length) {
        const c = upper[i];

        if (!std.ascii.isAlphabetic(c)) {
            i += 1;
            continue;
        }

        switch (c) {
            'A', 'E', 'I', 'O', 'U' => {
                // Vowels only encoded at the beginning
                if (i == 0 or (i > 0 and result.items.len == 0)) {
                    try result.append(c);
                }
                i += 1;
            },
            'B' => {
                // B not encoded if at end after M (e.g., "dumb")
                if (i == len - 1 and i > 0 and upper[i - 1] == 'M') {
                    // skip
                } else {
                    try result.append('B');
                }
                i += 1;
            },
            'C' => {
                // CH -> X
                if (i + 1 < len and upper[i + 1] == 'H') {
                    try result.append('X');
                    i += 2;
                }
                // CIA -> X
                else if (i + 2 < len and upper[i + 1] == 'I' and upper[i + 2] == 'A') {
                    try result.append('X');
                    i += 3;
                }
                // CE, CI, CY -> S
                else if (i + 1 < len and (upper[i + 1] == 'E' or upper[i + 1] == 'I' or upper[i + 1] == 'Y')) {
                    try result.append('S');
                    i += 2;
                }
                // Else -> K
                else {
                    try result.append('K');
                    i += 1;
                }
            },
            'D' => {
                // DGE, DGI, DGY -> J
                if (i + 2 < len and upper[i + 1] == 'G' and
                    (upper[i + 2] == 'E' or upper[i + 2] == 'I' or upper[i + 2] == 'Y'))
                {
                    try result.append('J');
                    i += 3;
                } else {
                    try result.append('T');
                    i += 1;
                }
            },
            'G' => {
                // GH and not at end or before vowel -> F
                if (i + 1 < len and upper[i + 1] == 'H') {
                    if (i + 2 < len and isVowel(upper[i + 2])) {
                        // silent
                        i += 2;
                    } else if (i > 0 and !isVowel(upper[i - 1])) {
                        try result.append('F');
                        i += 2;
                    } else {
                        i += 2;
                    }
                }
                // GN -> N
                else if (i + 1 < len and upper[i + 1] == 'N') {
                    try result.append('N');
                    i += 2;
                }
                // GE, GI, GY -> J
                else if (i + 1 < len and (upper[i + 1] == 'E' or upper[i + 1] == 'I' or upper[i + 1] == 'Y')) {
                    try result.append('J');
                    i += 2;
                } else {
                    try result.append('K');
                    i += 1;
                }
            },
            'H' => {
                // H silent after vowel and not before vowel
                if (i > 0 and isVowel(upper[i - 1]) and (i + 1 >= len or !isVowel(upper[i + 1]))) {
                    // silent
                } else {
                    try result.append('H');
                }
                i += 1;
            },
            'K' => {
                // CK -> K (but we only see K here after C was processed)
                if (i > 0 and upper[i - 1] == 'C') {
                    // skip duplicate
                } else {
                    try result.append('K');
                }
                i += 1;
            },
            'P' => {
                // PH -> F
                if (i + 1 < len and upper[i + 1] == 'H') {
                    try result.append('F');
                    i += 2;
                } else {
                    try result.append('P');
                    i += 1;
                }
            },
            'Q' => {
                try result.append('K');
                i += 1;
            },
            'S' => {
                // SH -> X
                if (i + 1 < len and upper[i + 1] == 'H') {
                    try result.append('X');
                    i += 2;
                }
                // SIO, SIA -> X
                else if (i + 2 < len and upper[i + 1] == 'I' and (upper[i + 2] == 'O' or upper[i + 2] == 'A')) {
                    try result.append('X');
                    i += 3;
                } else {
                    try result.append('S');
                    i += 1;
                }
            },
            'T' => {
                // TH -> 0 (theta)
                if (i + 1 < len and upper[i + 1] == 'H') {
                    try result.append('0');
                    i += 2;
                }
                // TIO, TIA -> X
                else if (i + 2 < len and upper[i + 1] == 'I' and (upper[i + 2] == 'O' or upper[i + 2] == 'A')) {
                    try result.append('X');
                    i += 3;
                }
                // TCH -> CH (but encode as X)
                else if (i + 2 < len and upper[i + 1] == 'C' and upper[i + 2] == 'H') {
                    try result.append('X');
                    i += 3;
                } else {
                    try result.append('T');
                    i += 1;
                }
            },
            'V' => {
                try result.append('F');
                i += 1;
            },
            'W', 'Y' => {
                // W, Y only encoded if followed by vowel
                if (i + 1 < len and isVowel(upper[i + 1])) {
                    try result.append(c);
                }
                i += 1;
            },
            'X' => {
                try result.append('K');
                try result.append('S');
                i += 1;
            },
            'Z' => {
                try result.append('S');
                i += 1;
            },
            else => {
                // F, J, L, M, N, R
                try result.append(c);
                i += 1;
            },
        }
    }

    return result.toOwnedSlice();
}

/// Check if two words match using Metaphone encoding
///
/// Time: O(n + m) where n, m are input lengths
/// Space: O(1) (plus encoding space)
pub fn metaphoneMatch(allocator: Allocator, word1: []const u8, word2: []const u8, max_length: usize) !bool {
    const code1 = try metaphone(allocator, word1, max_length);
    defer allocator.free(code1);
    const code2 = try metaphone(allocator, word2, max_length);
    defer allocator.free(code2);
    return std.mem.eql(u8, code1, code2);
}

/// Encode multiple words in batch
///
/// Returns a slice of encoded strings. Caller owns the returned slice and all strings.
/// Use freeBatch to clean up.
///
/// Time: O(k×n) where k is word count, n is average word length
/// Space: O(k×m) where m is max_length
pub fn metaphoneBatch(allocator: Allocator, words: []const []const u8, max_length: usize) ![][]u8 {
    const result = try allocator.alloc([]u8, words.len);
    errdefer {
        for (result, 0..) |code, idx| {
            if (idx < words.len) allocator.free(code);
        }
        allocator.free(result);
    }

    for (words, 0..) |word, i| {
        result[i] = try metaphone(allocator, word, max_length);
    }

    return result;
}

/// Free a batch of Metaphone codes
///
/// Time: O(k) where k is the number of codes
/// Space: O(1)
pub fn freeBatch(allocator: Allocator, codes: [][]u8) void {
    for (codes) |code| {
        allocator.free(code);
    }
    allocator.free(codes);
}

fn isVowel(c: u8) bool {
    return c == 'A' or c == 'E' or c == 'I' or c == 'O' or c == 'U';
}

// ============================================================================
// Tests
// ============================================================================

test "metaphone: basic examples" {
    const allocator = testing.allocator;

    // Classic examples from the paper
    const code1 = try metaphone(allocator, "knight", 4);
    defer allocator.free(code1);
    try testing.expectEqualStrings("NFT", code1);

    const code2 = try metaphone(allocator, "night", 4);
    defer allocator.free(code2);
    try testing.expectEqualStrings("NFT", code2);

    const code3 = try metaphone(allocator, "gnat", 4);
    defer allocator.free(code3);
    try testing.expectEqualStrings("NT", code3);

    const code4 = try metaphone(allocator, "gnaw", 4);
    defer allocator.free(code4);
    try testing.expectEqualStrings("N", code4);
}

test "metaphone: name matching" {
    const allocator = testing.allocator;

    // Smith variants
    const smith = try metaphone(allocator, "Smith", 4);
    defer allocator.free(smith);
    const smyth = try metaphone(allocator, "Smyth", 4);
    defer allocator.free(smyth);
    try testing.expectEqualStrings(smith, smyth);

    // Catherine variants
    const catherine = try metaphone(allocator, "Catherine", 4);
    defer allocator.free(catherine);
    const kathryn = try metaphone(allocator, "Kathryn", 4);
    defer allocator.free(kathryn);
    try testing.expectEqualStrings(catherine, kathryn);
}

test "metaphone: different names" {
    const allocator = testing.allocator;

    const john = try metaphone(allocator, "John", 4);
    defer allocator.free(john);
    const jane = try metaphone(allocator, "Jane", 4);
    defer allocator.free(jane);
    try testing.expect(!std.mem.eql(u8, john, jane));

    const robert = try metaphone(allocator, "Robert", 4);
    defer allocator.free(robert);
    const richard = try metaphone(allocator, "Richard", 4);
    defer allocator.free(richard);
    try testing.expect(!std.mem.eql(u8, robert, richard));
}

test "metaphone: edge cases" {
    const allocator = testing.allocator;

    // Empty string
    const empty = try metaphone(allocator, "", 4);
    defer allocator.free(empty);
    try testing.expectEqualStrings("", empty);

    // Single character
    const single = try metaphone(allocator, "A", 4);
    defer allocator.free(single);
    try testing.expectEqualStrings("A", single);

    // All vowels
    const vowels = try metaphone(allocator, "aeiou", 4);
    defer allocator.free(vowels);
    try testing.expectEqualStrings("A", vowels); // Only first vowel
}

test "metaphone: consonant clusters" {
    const allocator = testing.allocator;

    // PH -> F
    const phone = try metaphone(allocator, "phone", 4);
    defer allocator.free(phone);
    try testing.expect(phone[0] == 'F');

    // SH -> X
    const ship = try metaphone(allocator, "ship", 4);
    defer allocator.free(ship);
    try testing.expect(ship[0] == 'X');

    // CH -> X
    const church = try metaphone(allocator, "church", 4);
    defer allocator.free(church);
    try testing.expect(church[0] == 'X');

    // TH -> 0
    const think = try metaphone(allocator, "think", 4);
    defer allocator.free(think);
    try testing.expect(think[0] == '0');
}

test "metaphone: silent letters" {
    const allocator = testing.allocator;

    // Silent K in KN
    const knife = try metaphone(allocator, "knife", 4);
    defer allocator.free(knife);
    const nife = try metaphone(allocator, "nife", 4);
    defer allocator.free(nife);
    try testing.expectEqualStrings(knife, nife);

    // Silent G in GN
    const gnome = try metaphone(allocator, "gnome", 4);
    defer allocator.free(gnome);
    try testing.expect(gnome[0] == 'N');

    // Silent B in MB at end
    const dumb = try metaphone(allocator, "dumb", 4);
    defer allocator.free(dumb);
    const dum = try metaphone(allocator, "dum", 4);
    defer allocator.free(dum);
    try testing.expectEqualStrings(dumb, dum);
}

test "metaphone: max length constraint" {
    const allocator = testing.allocator;

    const long = try metaphone(allocator, "extraordinary", 4);
    defer allocator.free(long);
    try testing.expect(long.len <= 4);

    const short = try metaphone(allocator, "extraordinary", 2);
    defer allocator.free(short);
    try testing.expect(short.len <= 2);
}

test "metaphone: case insensitivity" {
    const allocator = testing.allocator;

    const lower = try metaphone(allocator, "hello", 4);
    defer allocator.free(lower);
    const upper = try metaphone(allocator, "HELLO", 4);
    defer allocator.free(upper);
    const mixed = try metaphone(allocator, "HeLLo", 4);
    defer allocator.free(mixed);

    try testing.expectEqualStrings(lower, upper);
    try testing.expectEqualStrings(lower, mixed);
}

test "metaphone: punctuation handling" {
    const allocator = testing.allocator;

    const plain = try metaphone(allocator, "hello", 4);
    defer allocator.free(plain);
    const punct = try metaphone(allocator, "hel-lo!", 4);
    defer allocator.free(punct);

    try testing.expectEqualStrings(plain, punct);
}

test "metaphone: match function" {
    const allocator = testing.allocator;

    try testing.expect(try metaphoneMatch(allocator, "knight", "night", 4));
    try testing.expect(try metaphoneMatch(allocator, "gnaw", "naw", 4));
    try testing.expect(!try metaphoneMatch(allocator, "cat", "dog", 4));
}

test "metaphone: batch encoding" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "knight", "night", "gnaw", "phone" };
    const codes = try metaphoneBatch(allocator, &words, 4);
    defer freeBatch(allocator, codes);

    try testing.expect(codes.len == 4);
    try testing.expectEqualStrings("NFT", codes[0]);
    try testing.expectEqualStrings("NFT", codes[1]);
}

test "metaphone: X replacement" {
    const allocator = testing.allocator;

    // Initial X -> S
    const xray = try metaphone(allocator, "xray", 4);
    defer allocator.free(xray);
    try testing.expect(xray[0] == 'S');

    // Internal X -> KS
    const fox = try metaphone(allocator, "fox", 4);
    defer allocator.free(fox);
    // F + KS (but limited by max_length)
    try testing.expect(fox[0] == 'F');
}

test "metaphone: soft C and G" {
    const allocator = testing.allocator;

    // C before E, I, Y -> S
    const city = try metaphone(allocator, "city", 4);
    defer allocator.free(city);
    try testing.expect(city[0] == 'S');

    const cent = try metaphone(allocator, "cent", 4);
    defer allocator.free(cent);
    try testing.expect(cent[0] == 'S');

    // G before E, I, Y -> J
    const gem = try metaphone(allocator, "gem", 4);
    defer allocator.free(gem);
    try testing.expect(gem[0] == 'J');

    const giant = try metaphone(allocator, "giant", 4);
    defer allocator.free(giant);
    try testing.expect(giant[0] == 'J');
}

test "metaphone: hard C and G" {
    const allocator = testing.allocator;

    // C not before E, I, Y -> K
    const cat = try metaphone(allocator, "cat", 4);
    defer allocator.free(cat);
    try testing.expect(cat[0] == 'K');

    // G not before E, I, Y -> K
    const go = try metaphone(allocator, "go", 4);
    defer allocator.free(go);
    try testing.expect(go[0] == 'K');
}

test "metaphone: vowel handling" {
    const allocator = testing.allocator;

    // Only initial vowel is encoded
    const apple = try metaphone(allocator, "apple", 4);
    defer allocator.free(apple);
    try testing.expect(apple[0] == 'A');

    const orange = try metaphone(allocator, "orange", 4);
    defer allocator.free(orange);
    try testing.expect(orange[0] == 'O');
}

test "metaphone: W and Y conditional" {
    const allocator = testing.allocator;

    // W before vowel
    const way = try metaphone(allocator, "way", 4);
    defer allocator.free(way);
    try testing.expect(way[0] == 'W');

    // Y before vowel
    const yes = try metaphone(allocator, "yes", 4);
    defer allocator.free(yes);
    try testing.expect(yes[0] == 'Y');
}

test "metaphone: DGE/DGI/DGY -> J" {
    const allocator = testing.allocator;

    const edge = try metaphone(allocator, "edge", 4);
    defer allocator.free(edge);
    try testing.expect(std.mem.indexOf(u8, edge, "J") != null);
}

test "metaphone: type variants" {
    const allocator = testing.allocator;

    const test1 = try metaphone(allocator, "test", 4);
    defer allocator.free(test1);
    try testing.expect(test1.len > 0);

    const test2 = try metaphone(allocator, "test", 6);
    defer allocator.free(test2);
    try testing.expect(test2.len > 0);
}

test "metaphone: memory safety" {
    const allocator = testing.allocator;

    // Multiple allocations and deallocations
    for (0..10) |_| {
        const code = try metaphone(allocator, "testing", 4);
        allocator.free(code);
    }
}

test "metaphone: real-world examples" {
    const allocator = testing.allocator;

    // Common misspellings that should match
    try testing.expect(try metaphoneMatch(allocator, "receive", "recieve", 4));
    try testing.expect(try metaphoneMatch(allocator, "separate", "seperate", 4));

    // Similar sounding words
    try testing.expect(try metaphoneMatch(allocator, "their", "there", 4));
    try testing.expect(try metaphoneMatch(allocator, "write", "right", 4));
}
