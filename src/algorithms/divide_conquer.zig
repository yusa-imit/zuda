/// Divide-and-Conquer algorithms for efficient problem solving.
///
/// Divide-and-conquer is an algorithmic paradigm that recursively breaks down
/// a problem into two or more sub-problems of the same or related type,
/// until these become simple enough to solve directly. The solutions to
/// the sub-problems are then combined to give a solution to the original problem.
///
/// ## Classic Problems
///
/// - **Quick Select**: Find kth smallest element in O(n) average time
/// - **Maximum Subarray**: Find contiguous subarray with largest sum
/// - **Closest Pair**: Find closest pair of points in 2D space
/// - **Binary Search**: Search in sorted array in O(log n) time
/// - **Merge Sort**: Sort in O(n log n) time (in sorting module)
///
/// ## Algorithm Pattern
///
/// ```zig
/// fn divideConquer(problem) {
///     if (problem is small enough) {
///         return solve directly;
///     }
///
///     // Divide
///     subproblems = split(problem);
///
///     // Conquer
///     for (subproblem in subproblems) {
///         subsolution = divideConquer(subproblem);
///     }
///
///     // Combine
///     return merge(subsolutions);
/// }
/// ```
///
/// ## Complexity Analysis
///
/// Most divide-and-conquer algorithms have logarithmic or linearithmic complexity:
/// - Binary Search: O(log n)
/// - Quick Select: O(n) average
/// - Merge Sort: O(n log n)
/// - Closest Pair: O(n log n)
///
/// ## Advantages
///
/// - Naturally parallel (sub-problems are independent)
/// - Efficient for large datasets
/// - Optimal for many fundamental problems
/// - Cache-friendly (works on smaller sub-problems)
///
/// ## Use Cases
///
/// - Sorting and searching
/// - Computational geometry
/// - Matrix operations
/// - Signal processing (FFT)
/// - Tree traversals

pub const quick_select = @import("divide_conquer/quick_select.zig");
pub const max_subarray = @import("divide_conquer/max_subarray.zig");
pub const closest_pair = @import("divide_conquer/closest_pair.zig");

// Re-export common functions
pub const quickSelect = quick_select.quickSelect;
pub const quickSelectBy = quick_select.quickSelectBy;
pub const median = quick_select.median;
pub const topK = quick_select.topK;

pub const kadane = max_subarray.kadane;
pub const maxSubarray = max_subarray.divideConquer;
pub const maxSum = max_subarray.maxSum;
pub const maxProduct = max_subarray.maxProduct;
pub const MaxSubarrayResult = max_subarray.Result;

pub const closestPair = closest_pair.closestPair;
pub const closestPairBruteForce = closest_pair.closestPairBruteForce;
pub const Point = closest_pair.Point;
pub const ClosestPairResult = closest_pair.ClosestPairResult;

test {
    @import("std").testing.refAllDecls(@This());
}
