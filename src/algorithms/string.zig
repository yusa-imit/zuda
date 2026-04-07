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
/// - **Longest Common Prefix**: O(n) find common prefix of string array
///   - Horizontal/vertical scanning, divide-and-conquer, binary search variants
///   - Suffix-based LCP for suffix array analysis
/// - **Suffix Array**: O(n log² n) construction, O(m log n) pattern search
///   - LCP array construction in O(n) using Kasai's algorithm
///   - Longest repeated substring, distinct substrings counting
/// - **Anagram Detection**: O(n) frequency-based anagram checking
///   - Character frequency matching, sorting-based comparison
///   - Sliding window anagram search, grouping by canonical form
///   - Case-insensitive and space-ignoring variants
///
/// ## Time Complexity
///
/// Most pattern matching algorithms achieve O(n+m) where n = text length, m = pattern length.
/// Manacher's algorithm is O(n) for palindrome detection.
/// Suffix array construction is O(n log² n), pattern search is O(m log n).
///
/// ## Space Complexity
///
/// Typically O(m) for pattern preprocessing, O(n) for full text analysis.
/// Suffix arrays require O(n) space for sorted suffix indices and auxiliary arrays.
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
pub const suffix_array = @import("string/suffix_array.zig");
pub const longest_common_prefix = @import("string/longest_common_prefix.zig");
pub const anagrams = @import("string/anagrams.zig");

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
pub const buildSuffixArray = suffix_array.buildSuffixArray;
pub const buildLCP = suffix_array.buildLCP;
pub const suffixArraySearch = suffix_array.search;
pub const longestRepeatedSubstring = suffix_array.longestRepeatedSubstring;
pub const countDistinctSubstrings = suffix_array.countDistinctSubstrings;
pub const SuffixArrayResult = suffix_array.SuffixArrayResult;
pub const LCPResult = suffix_array.LCPResult;
pub const longestCommonPrefix = longest_common_prefix.longestCommonPrefix;
pub const longestCommonPrefixVertical = longest_common_prefix.longestCommonPrefixVertical;
pub const longestCommonPrefixDivideConquer = longest_common_prefix.longestCommonPrefixDivideConquer;
pub const longestCommonPrefixBinarySearch = longest_common_prefix.longestCommonPrefixBinarySearch;
pub const findAllCommonPrefixLengths = longest_common_prefix.findAllCommonPrefixLengths;
pub const countStringsWithPrefix = longest_common_prefix.countStringsWithPrefix;
pub const longestCommonPrefixOfSuffixes = longest_common_prefix.longestCommonPrefixOfSuffixes;
pub const areAnagrams = anagrams.areAnagrams;
pub const areAnagramsSorted = anagrams.areAnagramsSorted;
pub const findAllAnagrams = anagrams.findAllAnagrams;
pub const groupAnagrams = anagrams.groupAnagrams;
pub const countAnagramPairs = anagrams.countAnagramPairs;
pub const areAnagramsIgnoreCaseAndSpaces = anagrams.areAnagramsIgnoreCaseAndSpaces;
pub const getCanonicalForm = anagrams.getCanonicalForm;

test {
    @import("std").testing.refAllDecls(@This());
}
