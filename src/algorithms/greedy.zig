/// Greedy Algorithms
///
/// Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum.
/// These algorithms are efficient but don't always guarantee optimal solutions for all problem variants.
///
/// Key characteristics:
/// - Make the locally best choice at each step
/// - Never reconsider previous choices (no backtracking)
/// - Often provide optimal solutions for specific problem classes
/// - Typically faster than dynamic programming (O(n log n) vs O(n²))
///
/// When to use greedy algorithms:
/// - Problem exhibits greedy choice property (local optimum leads to global optimum)
/// - Problem has optimal substructure
/// - Efficiency is critical and approximate solutions are acceptable
///
/// Classic applications:
/// - Activity selection (interval scheduling)
/// - Huffman coding (data compression)
/// - Fractional knapsack (resource allocation)
/// - Job sequencing (deadline scheduling)
/// - Minimum spanning trees (Kruskal's, Prim's)
/// - Shortest paths (Dijkstra's algorithm)

pub const activity_selection = @import("greedy/activity_selection.zig");
pub const huffman = @import("greedy/huffman.zig");
pub const knapsack = @import("greedy/knapsack.zig");
pub const job_sequencing = @import("greedy/job_sequencing.zig");
pub const coin_change = @import("greedy/coin_change.zig");

// Activity Selection
pub const Activity = activity_selection.Activity;
pub const activitySelection = activity_selection.activitySelection;
pub const weightedActivitySelection = activity_selection.weightedActivitySelection;

// Huffman Coding
pub const HuffmanNode = huffman.HuffmanNode;
pub const HuffmanCode = huffman.HuffmanCode;
pub const buildHuffmanTree = huffman.buildHuffmanTree;
pub const generateHuffmanCodes = huffman.generateHuffmanCodes;
pub const huffmanEncode = huffman.huffmanEncode;
pub const huffmanDecode = huffman.huffmanDecode;
pub const destroyHuffmanTree = huffman.destroyHuffmanTree;

// Knapsack
pub const Item = knapsack.Item;
pub const ItemSelection = knapsack.ItemSelection;
pub const fractionalKnapsack = knapsack.fractionalKnapsack;
pub const fractionalKnapsackDetailed = knapsack.fractionalKnapsackDetailed;
pub const zeroOneKnapsackGreedy = knapsack.zeroOneKnapsackGreedy;

// Job Sequencing
pub const Job = job_sequencing.Job;
pub const WeightedJob = job_sequencing.WeightedJob;
pub const JobSequence = job_sequencing.JobSequence;
pub const jobSequencing = job_sequencing.jobSequencing;
pub const jobSequencingWeighted = job_sequencing.jobSequencingWeighted;

// Coin Change
pub const CoinUsage = coin_change.CoinUsage;
pub const greedyCoinChange = coin_change.greedyCoinChange;
pub const greedyCoinChangeDetailed = coin_change.greedyCoinChangeDetailed;
pub const canMakeChange = coin_change.canMakeChange;
pub const minimumCoins = coin_change.minimumCoins;

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
