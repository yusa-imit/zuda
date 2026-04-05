/// String algorithms for pattern matching, manipulation, and analysis.
///
/// String processing is a fundamental area of computer science with applications
/// in text editors, search engines, bioinformatics, and data compression.
///
/// ## Pattern Matching
///
/// - **KMP (Knuth-Morris-Pratt)**: O(n+m) pattern matching with prefix function
/// - **Boyer-Moore**: O(n/m) average case with bad character/good suffix heuristics
/// - **Rabin-Karp**: O(n+m) rolling hash, good for multiple pattern search
/// - **Aho-Corasick**: O(n+m+z) multi-pattern matching, automaton-based
/// - **Z-Algorithm**: O(n) pattern matching via Z-array (longest common prefix)
///
/// ## String Analysis
///
/// - **Manacher's Algorithm**: O(n) longest palindromic substring
/// - **Glob Matching**: Wildcard pattern matching with * and ?
///
/// ## Time Complexity
///
/// Most pattern matching algorithms achieve O(n+m) where n = text length, m = pattern length.
/// Manacher's algorithm is O(n) for palindrome detection.
///
/// ## Space Complexity
///
/// Typically O(m) for pattern preprocessing, O(n) for full text analysis.
///
/// ## Use Cases
///
/// - Text search and replace
/// - DNA sequence analysis
/// - Data deduplication
/// - Spell checking and correction
/// - File path matching
/// - Palindrome detection

pub const kmp = @import("string/kmp.zig");
pub const boyer_moore = @import("string/boyer_moore.zig");
pub const rabin_karp = @import("string/rabin_karp.zig");
pub const aho_corasick = @import("string/aho_corasick.zig");
pub const z_algorithm = @import("string/z_algorithm.zig");
pub const glob_match = @import("string/glob_match.zig");
pub const manacher = @import("string/manacher.zig");

// Re-export common functions
pub const kmpSearch = kmp.search;
pub const kmpSearchAll = kmp.searchAll;
pub const boyerMooreSearch = boyer_moore.search;
pub const boyerMooreSearchAll = boyer_moore.searchAll;
pub const rabinKarpSearch = rabin_karp.search;
pub const rabinKarpSearchAll = rabin_karp.searchAll;
pub const ahoCorasickBuild = aho_corasick.build;
pub const ahoCorasickSearch = aho_corasick.search;
pub const zAlgorithm = z_algorithm.zAlgorithm;
pub const zSearch = z_algorithm.search;
pub const globMatch = glob_match.globMatch;
pub const longestPalindromicSubstring = manacher.longestPalindromicSubstring;
pub const longestPalindromeLength = manacher.longestPalindromeLength;
pub const countPalindromes = manacher.countPalindromes;
pub const allPalindromes = manacher.allPalindromes;
pub const PalindromeResult = manacher.PalindromeResult;

test {
    @import("std").testing.refAllDecls(@This());
}
