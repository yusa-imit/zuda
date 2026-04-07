//! Bitwise Algorithms
//!
//! This module provides efficient algorithms for bit manipulation operations,
//! Hamming distance computation, subset generation, and Gray code operations.

pub const bit_tricks = @import("bitwise/bit_tricks.zig");
pub const hamming = @import("bitwise/hamming.zig");
pub const subsets = @import("bitwise/subsets.zig");
pub const gray_code = @import("bitwise/gray_code.zig");

// Re-export commonly used functions for convenience
pub const popcount = bit_tricks.popcount;
pub const popcountFast = bit_tricks.popcountFast;
pub const countLeadingZeros = bit_tricks.countLeadingZeros;
pub const countTrailingZeros = bit_tricks.countTrailingZeros;
pub const isPowerOfTwo = bit_tricks.isPowerOfTwo;
pub const nextPowerOfTwo = bit_tricks.nextPowerOfTwo;
pub const parity = bit_tricks.parity;
pub const reverseBits = bit_tricks.reverseBits;
pub const binaryToGray = bit_tricks.binaryToGray;
pub const grayToBinary = bit_tricks.grayToBinary;
pub const swapBits = bit_tricks.swapBits;
pub const extractBitField = bit_tricks.extractBitField;
pub const setBitField = bit_tricks.setBitField;
pub const findMSB = bit_tricks.findMSB;
pub const findLSB = bit_tricks.findLSB;
pub const isolateRightmostBit = bit_tricks.isolateRightmostBit;
pub const clearRightmostBit = bit_tricks.clearRightmostBit;

// Hamming distance functions
pub const hammingDistance = hamming.hammingDistance;
pub const totalHammingDistance = hamming.totalHammingDistance;
pub const findNearestHamming = hamming.findNearestHamming;
pub const countWithinHammingRadius = hamming.countWithinHammingRadius;
pub const hammingDistanceBytes = hamming.hammingDistanceBytes;
pub const hammingDistanceBitwise = hamming.hammingDistanceBitwise;

// Subset generation
pub const SubsetIterator = subsets.SubsetIterator;
pub const SubsetOfSizeIterator = subsets.SubsetOfSizeIterator;
pub const SubmaskIterator = subsets.SubmaskIterator;
pub const subsetSize = subsets.subsetSize;
pub const isInSubset = subsets.isInSubset;
pub const generateAllSubsets = subsets.generateAllSubsets;
pub const generateSubsetsOfSize = subsets.generateSubsetsOfSize;

// Gray code generation and navigation
pub const generateGraySequence = gray_code.generateSequence;
pub const generateReflectedGray = gray_code.generateReflected;
pub const nextGray = gray_code.nextGray;
pub const previousGray = gray_code.previousGray;
pub const areAdjacentGray = gray_code.areAdjacent;
pub const changingBit = gray_code.changingBit;
pub const rankGray = gray_code.rank;
pub const unrankGray = gray_code.unrank;
