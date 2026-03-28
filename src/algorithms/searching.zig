//! Searching Algorithms
//!
//! This module provides efficient search algorithms for sorted and unsorted data.

pub const binary_search = @import("searching/binary_search.zig");
pub const interpolation_search = @import("searching/interpolation_search.zig");
pub const ternary_search = @import("searching/ternary_search.zig");

// Re-export commonly used functions for convenience
pub const binarySearch = binary_search.binarySearch;
pub const lowerBound = binary_search.lowerBound;
pub const upperBound = binary_search.upperBound;
pub const equalRange = binary_search.equalRange;
pub const exponentialSearch = binary_search.exponentialSearch;
pub const interpolationSearch = interpolation_search.interpolationSearch;
pub const ternarySearchMax = ternary_search.ternarySearchMax;
pub const ternarySearchMin = ternary_search.ternarySearchMin;
pub const ternarySearchMaxContinuous = ternary_search.ternarySearchMaxContinuous;
pub const ternarySearchMinContinuous = ternary_search.ternarySearchMinContinuous;
pub const TernaryResult = ternary_search.TernaryResult;
