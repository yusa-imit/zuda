/// Parallel algorithms for multi-threaded data processing
///
/// This module provides parallel patterns and algorithms for exploiting
/// multi-core parallelism. Includes sorting, map-reduce, and functional patterns.
///
/// Note: Current implementations use sequential execution as a baseline.
/// Real parallel versions would use std.Thread for task parallelism.
///
/// Categories:
/// - Parallel sorting (merge sort, quick sort)
/// - Map-reduce framework
/// - Parallel scan (prefix sum)
/// - Parallel reduction
/// - Parallel map/filter operations
/// - Group by and partition

pub const parallel_sort = @import("parallel/parallel_sort.zig");
pub const map_reduce = @import("parallel/map_reduce.zig");

// Re-export commonly used functions
pub const parallelMergeSort = parallel_sort.parallelMergeSort;
pub const parallelQuickSort = parallel_sort.parallelQuickSort;
pub const parallelPrefixSum = parallel_sort.parallelPrefixSum;
pub const parallelReduce = parallel_sort.parallelReduce;
pub const parallelMap = parallel_sort.parallelMap;
pub const parallelFilter = parallel_sort.parallelFilter;

pub const mapReduce = map_reduce.mapReduce;
pub const MapReduceResult = map_reduce.MapReduceResult;
pub const groupBy = map_reduce.groupBy;
pub const GroupByResult = map_reduce.GroupByResult;
pub const partition = map_reduce.partition;
pub const PartitionResult = map_reduce.PartitionResult;
