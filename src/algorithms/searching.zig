//! Searching Algorithms
//!
//! This module provides efficient search algorithms for sorted and unsorted data.

pub const binary_search = @import("searching/binary_search.zig");

// Re-export commonly used functions for convenience
pub const binarySearch = binary_search.binarySearch;
pub const lowerBound = binary_search.lowerBound;
pub const upperBound = binary_search.upperBound;
pub const equalRange = binary_search.equalRange;
pub const exponentialSearch = binary_search.exponentialSearch;
