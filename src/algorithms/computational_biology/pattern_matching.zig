const std = @import("std");
const Allocator = std.mem.Allocator;

/// Find all occurrences of a pattern in a sequence using KMP algorithm
///
/// **Algorithm**: Knuth-Morris-Pratt with failure function
/// **Time**: O(n + m) where n = text length, m = pattern length
/// **Space**: O(m) for failure function
///
/// **Use cases**:
/// - Finding motifs in DNA sequences
/// - Searching for protein domains
/// - Restriction site detection
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `text`: Text sequence to search in
/// - `pattern`: Pattern to search for
///
/// **Returns**: ArrayList of starting positions where pattern occurs
pub fn findPattern(
    allocator: Allocator,
    text: []const u8,
    pattern: []const u8,
) !std.ArrayList(usize) {
    var matches = std.ArrayList(usize).init(allocator);
    errdefer matches.deinit();

    if (pattern.len == 0 or pattern.len > text.len) {
        return matches;
    }

    // Build failure function (LPS array)
    const lps = try computeLPS(allocator, pattern);
    defer allocator.free(lps);

    // KMP search
    var i: usize = 0; // text index
    var j: usize = 0; // pattern index

    while (i < text.len) {
        if (text[i] == pattern[j]) {
            i += 1;
            j += 1;
        }

        if (j == pattern.len) {
            // Found match
            try matches.append(i - j);
            j = lps[j - 1];
        } else if (i < text.len and text[i] != pattern[j]) {
            // Mismatch after j matches
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
    }

    return matches;
}

/// Compute LPS (Longest Proper Prefix which is also Suffix) array for KMP
fn computeLPS(allocator: Allocator, pattern: []const u8) ![]usize {
    const m = pattern.len;
    const lps = try allocator.alloc(usize, m);
    errdefer allocator.free(lps);

    lps[0] = 0;
    var len: usize = 0;
    var i: usize = 1;

    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len += 1;
            lps[i] = len;
            i += 1;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i += 1;
            }
        }
    }

    return lps;
}

/// Find reverse complement of a DNA sequence
///
/// **Time**: O(n) where n = sequence length
/// **Space**: O(n) for result
///
/// **Use cases**:
/// - Finding palindromic restriction sites
/// - Searching both strands of DNA
/// - Primer design
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `seq`: DNA sequence (must contain only A, T, C, G)
///
/// **Returns**: Reverse complement sequence
/// **Errors**: `InvalidBase` if sequence contains non-DNA characters
pub fn reverseComplement(allocator: Allocator, seq: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, seq.len);
    errdefer allocator.free(result);

    var i: usize = 0;
    while (i < seq.len) : (i += 1) {
        const base = seq[seq.len - 1 - i];
        result[i] = switch (base) {
            'A' => 'T',
            'T' => 'A',
            'C' => 'G',
            'G' => 'C',
            'a' => 't',
            't' => 'a',
            'c' => 'g',
            'g' => 'c',
            else => return error.InvalidBase,
        };
    }

    return result;
}

/// Count GC content (percentage of G and C bases)
///
/// **Time**: O(n) where n = sequence length
/// **Space**: O(1)
///
/// **Use cases**:
/// - Gene prediction
/// - Primer design (melting temperature)
/// - Species identification
///
/// **Parameters**:
/// - `seq`: DNA or RNA sequence
///
/// **Returns**: GC content as percentage (0.0 to 100.0)
pub fn gcContent(seq: []const u8) f64 {
    if (seq.len == 0) return 0.0;

    var gc_count: usize = 0;
    for (seq) |base| {
        switch (base) {
            'G', 'C', 'g', 'c' => gc_count += 1,
            else => {},
        }
    }

    return @as(f64, @floatFromInt(gc_count)) / @as(f64, @floatFromInt(seq.len)) * 100.0;
}

/// Transcribe DNA to RNA (T -> U)
///
/// **Time**: O(n) where n = sequence length
/// **Space**: O(n) for result
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `dna`: DNA sequence
///
/// **Returns**: RNA sequence
pub fn transcribe(allocator: Allocator, dna: []const u8) ![]u8 {
    const rna = try allocator.alloc(u8, dna.len);
    errdefer allocator.free(rna);

    for (dna, 0..) |base, i| {
        rna[i] = switch (base) {
            'T' => 'U',
            't' => 'u',
            else => base,
        };
    }

    return rna;
}

/// Translate RNA to protein sequence using standard genetic code
///
/// **Time**: O(n/3) where n = RNA sequence length
/// **Space**: O(n/3) for result
///
/// **Use cases**:
/// - Gene annotation
/// - Protein prediction
/// - Open reading frame (ORF) finding
///
/// **Parameters**:
/// - `allocator`: Memory allocator
/// - `rna`: RNA sequence (length should be multiple of 3)
///
/// **Returns**: Protein sequence (single letter amino acid codes)
/// **Note**: Stops at first stop codon (UAA, UAG, UGA)
pub fn translate(allocator: Allocator, rna: []const u8) ![]u8 {
    var protein = std.ArrayList(u8).init(allocator);
    errdefer protein.deinit();

    var i: usize = 0;
    while (i + 2 < rna.len) : (i += 3) {
        const codon = rna[i .. i + 3];
        const amino_acid = codonToAminoAcid(codon) orelse return error.InvalidCodon;

        if (amino_acid == '*') break; // Stop codon

        try protein.append(amino_acid);
    }

    return protein.toOwnedSlice();
}

/// Standard genetic code lookup
fn codonToAminoAcid(codon: []const u8) ?u8 {
    if (codon.len != 3) return null;

    // Standard genetic code (simplified)
    const code = std.StaticStringMap(u8).initComptime(.{
        .{ "UUU", 'F' }, .{ "UUC", 'F' }, .{ "UUA", 'L' }, .{ "UUG", 'L' },
        .{ "UCU", 'S' }, .{ "UCC", 'S' }, .{ "UCA", 'S' }, .{ "UCG", 'S' },
        .{ "UAU", 'Y' }, .{ "UAC", 'Y' }, .{ "UAA", '*' }, .{ "UAG", '*' },
        .{ "UGU", 'C' }, .{ "UGC", 'C' }, .{ "UGA", '*' }, .{ "UGG", 'W' },
        .{ "CUU", 'L' }, .{ "CUC", 'L' }, .{ "CUA", 'L' }, .{ "CUG", 'L' },
        .{ "CCU", 'P' }, .{ "CCC", 'P' }, .{ "CCA", 'P' }, .{ "CCG", 'P' },
        .{ "CAU", 'H' }, .{ "CAC", 'H' }, .{ "CAA", 'Q' }, .{ "CAG", 'Q' },
        .{ "CGU", 'R' }, .{ "CGC", 'R' }, .{ "CGA", 'R' }, .{ "CGG", 'R' },
        .{ "AUU", 'I' }, .{ "AUC", 'I' }, .{ "AUA", 'I' }, .{ "AUG", 'M' },
        .{ "ACU", 'T' }, .{ "ACC", 'T' }, .{ "ACA", 'T' }, .{ "ACG", 'T' },
        .{ "AAU", 'N' }, .{ "AAC", 'N' }, .{ "AAA", 'K' }, .{ "AAG", 'K' },
        .{ "AGU", 'S' }, .{ "AGC", 'S' }, .{ "AGA", 'R' }, .{ "AGG", 'R' },
        .{ "GUU", 'V' }, .{ "GUC", 'V' }, .{ "GUA", 'V' }, .{ "GUG", 'V' },
        .{ "GCU", 'A' }, .{ "GCC", 'A' }, .{ "GCA", 'A' }, .{ "GCG", 'A' },
        .{ "GAU", 'D' }, .{ "GAC", 'D' }, .{ "GAA", 'E' }, .{ "GAG", 'E' },
        .{ "GGU", 'G' }, .{ "GGC", 'G' }, .{ "GGA", 'G' }, .{ "GGG", 'G' },
    });

    return code.get(codon);
}

// ============================================================================
// Tests
// ============================================================================

test "findPattern: single occurrence" {
    const allocator = std.testing.allocator;
    const text = "ACGTACGTACGT";
    const pattern = "TACG";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 2), matches.items.len);
    try std.testing.expectEqual(@as(usize, 2), matches.items[0]);
    try std.testing.expectEqual(@as(usize, 6), matches.items[1]);
}

test "findPattern: no matches" {
    const allocator = std.testing.allocator;
    const text = "AAAA";
    const pattern = "TTT";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 0), matches.items.len);
}

test "findPattern: overlapping" {
    const allocator = std.testing.allocator;
    const text = "AAAA";
    const pattern = "AA";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 3), matches.items.len);
    try std.testing.expectEqual(@as(usize, 0), matches.items[0]);
    try std.testing.expectEqual(@as(usize, 1), matches.items[1]);
    try std.testing.expectEqual(@as(usize, 2), matches.items[2]);
}

test "findPattern: empty pattern" {
    const allocator = std.testing.allocator;
    const text = "ACGT";

    var matches = try findPattern(allocator, text, "");
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 0), matches.items.len);
}

test "findPattern: pattern longer than text" {
    const allocator = std.testing.allocator;
    const text = "ACG";
    const pattern = "ACGTACGT";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 0), matches.items.len);
}

test "findPattern: entire text" {
    const allocator = std.testing.allocator;
    const text = "ACGT";
    const pattern = "ACGT";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 1), matches.items.len);
    try std.testing.expectEqual(@as(usize, 0), matches.items[0]);
}

test "reverseComplement: basic" {
    const allocator = std.testing.allocator;
    const seq = "ACGT";

    const rc = try reverseComplement(allocator, seq);
    defer allocator.free(rc);

    try std.testing.expectEqualStrings("ACGT", rc);
}

test "reverseComplement: palindrome" {
    const allocator = std.testing.allocator;
    const seq = "GAATTC"; // EcoRI restriction site

    const rc = try reverseComplement(allocator, seq);
    defer allocator.free(rc);

    try std.testing.expectEqualStrings("GAATTC", rc);
}

test "reverseComplement: lowercase" {
    const allocator = std.testing.allocator;
    const seq = "acgt";

    const rc = try reverseComplement(allocator, seq);
    defer allocator.free(rc);

    try std.testing.expectEqualStrings("acgt", rc);
}

test "reverseComplement: invalid base" {
    const allocator = std.testing.allocator;
    const seq = "ACGX";

    try std.testing.expectError(error.InvalidBase, reverseComplement(allocator, seq));
}

test "reverseComplement: empty" {
    const allocator = std.testing.allocator;
    const seq = "";

    const rc = try reverseComplement(allocator, seq);
    defer allocator.free(rc);

    try std.testing.expectEqual(@as(usize, 0), rc.len);
}

test "gcContent: balanced" {
    const seq = "ACGT";
    const gc = gcContent(seq);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), gc, 0.01);
}

test "gcContent: all GC" {
    const seq = "GCGC";
    const gc = gcContent(seq);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), gc, 0.01);
}

test "gcContent: no GC" {
    const seq = "ATAT";
    const gc = gcContent(seq);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gc, 0.01);
}

test "gcContent: empty" {
    const seq = "";
    const gc = gcContent(seq);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gc, 0.01);
}

test "gcContent: lowercase" {
    const seq = "acgt";
    const gc = gcContent(seq);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), gc, 0.01);
}

test "transcribe: basic" {
    const allocator = std.testing.allocator;
    const dna = "ACGT";

    const rna = try transcribe(allocator, dna);
    defer allocator.free(rna);

    try std.testing.expectEqualStrings("ACGU", rna);
}

test "transcribe: lowercase" {
    const allocator = std.testing.allocator;
    const dna = "acgt";

    const rna = try transcribe(allocator, dna);
    defer allocator.free(rna);

    try std.testing.expectEqualStrings("acgu", rna);
}

test "transcribe: empty" {
    const allocator = std.testing.allocator;
    const dna = "";

    const rna = try transcribe(allocator, dna);
    defer allocator.free(rna);

    try std.testing.expectEqual(@as(usize, 0), rna.len);
}

test "translate: start codon" {
    const allocator = std.testing.allocator;
    const rna = "AUGUUUUAG"; // AUG (M) UUU (F) UAG (stop)

    const protein = try translate(allocator, rna);
    defer allocator.free(protein);

    try std.testing.expectEqualStrings("MF", protein);
}

test "translate: no stop codon" {
    const allocator = std.testing.allocator;
    const rna = "AUGUUU"; // AUG (M) UUU (F)

    const protein = try translate(allocator, rna);
    defer allocator.free(protein);

    try std.testing.expectEqualStrings("MF", protein);
}

test "translate: immediate stop" {
    const allocator = std.testing.allocator;
    const rna = "UAA"; // Stop codon

    const protein = try translate(allocator, rna);
    defer allocator.free(protein);

    try std.testing.expectEqual(@as(usize, 0), protein.len);
}

test "translate: empty" {
    const allocator = std.testing.allocator;
    const rna = "";

    const protein = try translate(allocator, rna);
    defer allocator.free(protein);

    try std.testing.expectEqual(@as(usize, 0), protein.len);
}

test "translate: all amino acids" {
    const allocator = std.testing.allocator;
    // Sample codons for various amino acids
    const rna = "UUUUUCUUAUUGUCUUCCUCAUCG"; // F F L L S S S S

    const protein = try translate(allocator, rna);
    defer allocator.free(protein);

    try std.testing.expectEqualStrings("FFLLSSSS", protein);
}

test "findPattern: restriction site" {
    const allocator = std.testing.allocator;
    const text = "ACGAATTCGAATTCTA"; // Contains two EcoRI sites
    const pattern = "GAATTC";

    var matches = try findPattern(allocator, text, pattern);
    defer matches.deinit();

    try std.testing.expectEqual(@as(usize, 2), matches.items.len);
    try std.testing.expectEqual(@as(usize, 3), matches.items[0]);
    try std.testing.expectEqual(@as(usize, 9), matches.items[1]);
}
