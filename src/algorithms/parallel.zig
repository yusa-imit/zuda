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
pub const prefix_sum = @import("parallel/prefix_sum.zig");

// Re-export commonly used functions
pub const parallelMergeSort = parallel_sort.parallelMergeSort;
pub const parallelQuickSort = parallel_sort.parallelQuickSort;
pub const parallelMap = parallel_sort.parallelMap;
pub const parallelFilter = parallel_sort.parallelFilter;

// Prefix sum operations (from dedicated module)
pub const inclusiveScan = prefix_sum.inclusiveScan;
pub const exclusiveScan = prefix_sum.exclusiveScan;
pub const parallelScan = prefix_sum.parallelScan;
pub const scanInPlace = prefix_sum.scanInPlace;
pub const segmentedScan = prefix_sum.segmentedScan;
pub const reduce = prefix_sum.reduce;

// Legacy compatibility - use exclusiveScan instead
pub const parallelPrefixSum = prefix_sum.exclusiveScan;
pub const parallelReduce = prefix_sum.reduce;

pub const mapReduce = map_reduce.mapReduce;
pub const MapReduceResult = map_reduce.MapReduceResult;
pub const groupBy = map_reduce.groupBy;
pub const GroupByResult = map_reduce.GroupByResult;
pub const partition = map_reduce.partition;
pub const PartitionResult = map_reduce.PartitionResult;
