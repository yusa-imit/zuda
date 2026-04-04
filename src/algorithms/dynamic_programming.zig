// Dynamic Programming Algorithms
//
// A collection of classic dynamic programming algorithms for optimization,
// counting, and string matching problems.

pub const binary_search = @import("dynamic_programming/binary_search.zig");
pub const climbing_stairs = @import("dynamic_programming/climbing_stairs.zig");
pub const coin_change = @import("dynamic_programming/coin_change.zig");
pub const distinct_subsequences = @import("dynamic_programming/distinct_subsequences.zig");
pub const edit_distance = @import("dynamic_programming/edit_distance.zig");
pub const egg_drop = @import("dynamic_programming/egg_drop.zig");
pub const house_robber = @import("dynamic_programming/house_robber.zig");
pub const knapsack = @import("dynamic_programming/knapsack.zig");
pub const lcs = @import("dynamic_programming/lcs.zig");
pub const lis = @import("dynamic_programming/lis.zig");
pub const longest_common_substring = @import("dynamic_programming/longest_common_substring.zig");
pub const lps = @import("dynamic_programming/lps.zig");
pub const matrix_chain = @import("dynamic_programming/matrix_chain.zig");
pub const max_product_subarray = @import("dynamic_programming/max_product_subarray.zig");
pub const max_sum_subarray = @import("dynamic_programming/max_sum_subarray.zig");
pub const palindrome_partition = @import("dynamic_programming/palindrome_partition.zig");
pub const regex_matching = @import("dynamic_programming/regex_matching.zig");
pub const rod_cutting = @import("dynamic_programming/rod_cutting.zig");
pub const subset_sum = @import("dynamic_programming/subset_sum.zig");
pub const unique_paths = @import("dynamic_programming/unique_paths.zig");
pub const wildcard_matching = @import("dynamic_programming/wildcard_matching.zig");
pub const word_break = @import("dynamic_programming/word_break.zig");

// Re-export commonly used functions
pub const longestIncreasingSubsequence = lis.longestIncreasingSubsequence;
pub const longestCommonSubsequence = lcs.longestCommonSubsequence;
pub const editDistance = edit_distance.editDistance;
pub const knapsack01 = knapsack.knapsack01;
pub const coinChange = coin_change.minCoins;
pub const isMatch = wildcard_matching.isMatch;
pub const isMatchOptimized = wildcard_matching.isMatchOptimized;
pub const isMatchGreedy = wildcard_matching.isMatchGreedy;
pub const regexMatch = regex_matching.isMatch;
pub const regexMatchOptimized = regex_matching.isMatchOptimized;
pub const regexMatchRecursive = regex_matching.isMatchRecursive;
