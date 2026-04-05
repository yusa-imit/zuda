/// Backtracking algorithms for systematic exploration of solution spaces.
///
/// Backtracking is a general algorithmic technique for finding all (or some) solutions
/// to computational problems that incrementally builds candidates and abandons a candidate
/// ("backtracks") as soon as it determines that the candidate cannot lead to a valid solution.
///
/// ## Classic Problems
///
/// - **N-Queens**: Place N queens on NxN board with no attacks
/// - **Sudoku**: Fill 9x9 grid with digits 1-9 following rules
/// - **Permutations**: Generate all orderings of elements
/// - **Subsets**: Generate all subsets (power set)
/// - **Combination Sum**: Find combinations summing to target
/// - **Word Search**: Find word in 2D character grid with DFS
///
/// ## Algorithm Pattern
///
/// ```zig
/// fn backtrack(state, ...) {
///     if (is_solution(state)) {
///         record_solution(state);
///         return;
///     }
///
///     for (choice in choices) {
///         if (is_valid(choice)) {
///             make_choice(choice);
///             backtrack(new_state, ...);
///             undo_choice(choice);  // Backtrack
///         }
///     }
/// }
/// ```
///
/// ## Time Complexity
///
/// Most backtracking algorithms have exponential time complexity:
/// - N-Queens: O(N!)
/// - Sudoku: O(9^(n*n)) with heavy pruning
/// - Permutations: O(N! * N)
/// - Subsets: O(N * 2^N)
///
/// ## Space Complexity
///
/// Typically O(N) for recursion stack + result storage.

pub const n_queens = @import("backtracking/n_queens.zig");
pub const sudoku = @import("backtracking/sudoku.zig");
pub const permutations = @import("backtracking/permutations.zig");
pub const subsets = @import("backtracking/subsets.zig");
pub const combination_sum = @import("backtracking/combination_sum.zig");
pub const word_search = @import("backtracking/word_search.zig");

// Re-export common functions
pub const solveNQueens = n_queens.solveNQueens;
pub const countNQueens = n_queens.countNQueens;
pub const solveSudoku = sudoku.solveSudoku;
pub const isValidSudoku = sudoku.isValidSudoku;
pub const permute = permutations.permute;
pub const permuteUnique = permutations.permuteUnique;
pub const subsetsAll = subsets.subsets;
pub const subsetsOfSize = subsets.subsetsOfSize;
pub const subsetsUnique = subsets.subsetsUnique;
pub const combinationSum = combination_sum.combinationSum;
pub const combinationSumUnique = combination_sum.combinationSumUnique;
pub const exist = word_search.exist;
pub const existWithPath = word_search.existWithPath;
pub const findAll = word_search.findAll;
pub const countOccurrences = word_search.countOccurrences;
pub const Position = word_search.Position;

test {
    @import("std").testing.refAllDecls(@This());
}
