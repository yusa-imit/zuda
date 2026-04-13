/// Combinatorics Algorithms
///
/// This module provides fundamental combinatorial functions and algorithms.
///
/// ## Algorithms Overview
///
/// ### Factorial
/// **Use Case**: Counting permutations, probability calculations
/// - **Time**: O(n)
/// - **Space**: O(1)
/// - **Properties**: n! grows very fast, overflow detection needed
///
/// ### Binomial Coefficients (n choose k)
/// **Use Case**: Counting combinations, probability theory
/// - **Time**: O(k) with multiplicative formula
/// - **Space**: O(1)
/// - **Properties**: Symmetric C(n, k) = C(n, n-k), Pascal's triangle
///
/// ### Permutations (n P k)
/// **Use Case**: Counting arrangements, ordered selections
/// - **Time**: O(k)
/// - **Space**: O(1)
/// - **Properties**: P(n, k) = n! / (n-k)!
///
/// ### Catalan Numbers
/// **Use Case**: Counting trees, parentheses, paths
/// - **Time**: O(n)
/// - **Space**: O(1)
/// - **Applications**: Binary trees, valid parentheses, Dyck paths
///
/// ### Stirling Numbers
/// **Use Case**: Partitioning sets, combinatorial identities
/// - **Time**: O(n × k) for S(n, k)
/// - **Space**: O(k) with DP
/// - **Types**: First kind (cycles), second kind (partitions)
///
/// ### Permutation Generation
/// **Use Case**: Enumerating all arrangements
/// - **Time**: O(n! × n)
/// - **Space**: O(n!)
/// - **Algorithm**: Heap's algorithm via backtracking
///
/// ### Combination Generation
/// **Use Case**: Enumerating all selections
/// - **Time**: O(C(n, k) × k)
/// - **Space**: O(C(n, k))
/// - **Algorithm**: Recursive generation
///
/// ### Integer Partitions
/// **Use Case**: Partitioning numbers, dynamic programming
/// - **Time**: O(n×k) for counting, O(p(n)) for generation
/// - **Space**: O(k) for counting, O(p(n)) for generation
/// - **Applications**: Number theory, partition function, subset sum
///
/// ### Bell Numbers
/// **Use Case**: Counting set partitions
/// - **Time**: O(n²)
/// - **Space**: O(n)
/// - **Applications**: Set theory, equivalence relations
///
/// ### Derangements
/// **Use Case**: Permutations with no fixed points
/// - **Time**: O(n)
/// - **Space**: O(1)
/// - **Applications**: Combinatorial probability, hatcheck problem
///
/// ## Example
///
/// ```zig
/// const combinatorics = @import("zuda").algorithms.combinatorics;
///
/// // Counting
/// const fact5 = try combinatorics.basics.factorial(u64, 5); // 120
/// const c_10_3 = try combinatorics.basics.binomial(u64, 10, 3); // 120
/// const p_10_3 = try combinatorics.basics.permutation(u64, 10, 3); // 720
///
/// // Generation
/// const items = [_]u8{1, 2, 3};
/// const perms = try combinatorics.basics.generatePermutations(u8, allocator, &items);
/// defer {
///     for (perms.items) |perm| allocator.free(perm);
///     perms.deinit();
/// }
///
/// // Partitions
/// const part_count = try combinatorics.partitions.countPartitions(u32, 5, 2); // 2
/// const bell5 = try combinatorics.partitions.bellNumber(u32, 5); // 52
/// const derang3 = try combinatorics.partitions.derangements(u32, 3); // 2
/// ```

pub const basics = @import("combinatorics/basics.zig");
pub const partitions = @import("combinatorics/partitions.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
