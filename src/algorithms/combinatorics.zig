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
/// ### Integer Compositions
/// **Use Case**: Ordered partitions, sequence analysis
/// - **Time**: O(C(n-1, k-1)) for k-compositions, O(2^(n-1)) for all
/// - **Space**: O(C(n-1, k-1) × k) for generation
/// - **Applications**: Resource allocation (ordered), dynamic programming, probability theory
///
/// ### Integer Sequences
/// **Use Case**: Fundamental sequences in combinatorics and number theory
/// - **Time**: O(n) for Fibonacci/Lucas/Pell, O(1) for closed-form (triangular, pentagonal, etc.)
/// - **Space**: O(1) for single values, O(n) for sequence generation
/// - **Sequences**: Fibonacci, Lucas, Triangular, Pentagonal, Square, Tetrahedral, Pell, Polygonal
/// - **Applications**: Recurrence relations, number theory, combinatorial identities, golden ratio approximations
///
/// ### Lexicographic Permutation Operations
/// **Use Case**: Efficient permutation enumeration, algorithm competitions
/// - **Time**: O(n) for next/prev, O(n²) for k-th permutation and rank
/// - **Space**: O(1) for next/prev, O(n) for k-th permutation and rank
/// - **Operations**: nextPermutation, prevPermutation, kthPermutation, permutationRank
/// - **Applications**: Permutation generation, combinatorial search, testing, optimization
///
/// ### Young Tableaux
/// **Use Case**: Representation theory, symmetric functions, algebraic combinatorics
/// - **Time**: O(n) for hook length formula, O(n²) for Robinson-Schensted, O(SYT(λ) × n) for generation
/// - **Space**: O(n) for tableau storage and operations
/// - **Operations**: countStandardTableaux (hook formula), isStandard (validation), robinsonSchenstedP (permutation bijection)
/// - **Applications**: Symmetric group representations, Schur functions, Robinson-Schensted correspondence, plactic monoid
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
///
/// // Compositions
/// const comp_count = try combinatorics.compositions.countCompositions(u32, 5, 2); // 4
/// const all_comps = try combinatorics.compositions.countAllCompositions(u32, 4); // 8
/// const k_comps = try combinatorics.compositions.generateKCompositions(u32, allocator, 5, 2);
/// defer {
///     for (k_comps.items) |c| allocator.free(c);
///     k_comps.deinit();
/// }
///
/// // Stirling Numbers
/// const S_4_2 = try combinatorics.stirling.stirlingFirst(u64, 4, 2, allocator); // 11
/// const s_4_2 = try combinatorics.stirling.stirlingSecond(u64, 4, 2, allocator); // 7
/// const row = try combinatorics.stirling.generateStirlingSecondRow(u64, 4, allocator);
/// defer allocator.free(row);
///
/// // Integer Sequences
/// const fib10 = combinatorics.sequences.fibonacci(u64, 10); // 55
/// const lucas7 = combinatorics.sequences.lucas(u64, 7); // 29
/// const tri5 = combinatorics.sequences.triangular(u64, 5); // 15
/// const pent4 = combinatorics.sequences.pentagonal(u64, 4); // 22
/// const pell5 = combinatorics.sequences.pell(u64, 5); // 29
/// const hex3 = combinatorics.sequences.polygonal(u64, 6, 3); // 15 (hexagonal)
/// const fib_seq = try combinatorics.sequences.generateFibonacci(u64, allocator, 10);
/// defer fib_seq.deinit();
///
/// // Lexicographic Permutations
/// var perm = [_]u8{0, 1, 2};
/// while (combinatorics.permutations.nextPermutation(u8, &perm)) {
///     // Process perm
/// }
/// const kth = try combinatorics.permutations.kthPermutation(u32, allocator, 5, 42);
/// defer allocator.free(kth);
/// const rank = try combinatorics.permutations.permutationRank(u32, allocator, &[_]u32{2, 0, 1});
///
/// // Young Tableaux
/// const shape = [_]usize{3, 2, 1}; // partition λ = (3,2,1)
/// const syt_count = try combinatorics.young_tableaux.countStandardTableaux(u64, &shape); // 16
/// const perm = [_]u32{3, 1, 4, 2};
/// const tab = try combinatorics.young_tableaux.robinsonSchenstedP(allocator, &perm);
/// defer tab.deinit();
/// const is_std = try tab.isStandard(allocator); // true
/// ```

pub const basics = @import("combinatorics/basics.zig");
pub const partitions = @import("combinatorics/partitions.zig");
pub const compositions = @import("combinatorics/compositions.zig");
pub const stirling = @import("combinatorics/stirling.zig");
pub const sequences = @import("combinatorics/sequences.zig");
pub const permutations = @import("combinatorics/permutations.zig");
pub const catalan = @import("combinatorics/catalan.zig");
pub const young_tableaux = @import("combinatorics/young_tableaux.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
