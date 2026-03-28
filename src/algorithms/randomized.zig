/// Randomized Algorithms
///
/// This module provides probabilistic and randomized algorithms that use
/// random number generation to achieve efficient average-case performance
/// or to solve problems where deterministic solutions are impractical.
///
/// Randomized algorithms fall into two categories:
///
/// 1. **Las Vegas algorithms**: Always produce correct results, but running
///    time is random (e.g., randomized quickselect)
///
/// 2. **Monte Carlo algorithms**: Running time is deterministic or bounded,
///    but may produce incorrect results with small probability (e.g.,
///    Miller-Rabin primality test)
///
/// ## Available Algorithms
///
/// ### Fisher-Yates Shuffle
/// - `shuffle()`: In-place random permutation - O(n) time, O(1) space
/// - `partialShuffle()`: Random k-subset selection - O(k) time
/// - `randomPermutation()`: Generate random permutation - O(n) time, O(n) space
///
/// Properties:
/// - Unbiased: all n! permutations equally likely
/// - Single pass through array
/// - Optimal for random shuffling (cards, sampling, etc.)
///
/// ### Reservoir Sampling
/// - `reservoirSample()`: Uniform random k-sample from stream - O(n) time, O(k) space
/// - `weightedReservoirSample()`: Weighted random sampling - O(n log k) time
/// - `combineReservoirs()`: Merge distributed samples - O(n) time
///
/// Properties:
/// - Single pass over data (streaming algorithm)
/// - Unknown or infinite stream size
/// - Each element has probability k/n of selection
///
/// Use cases:
/// - Random sampling from large files
/// - Online sampling from data streams
/// - Distributed sampling across multiple sources
///
/// ### Randomized Selection (Quickselect)
/// - `randomizedSelect()`: Find kth smallest element - O(n) expected, O(n²) worst
/// - `median()`: Find median - O(n) expected time
/// - `topK()`: Find k largest elements - O(n) expected time
///
/// Properties:
/// - Expected linear time with high probability
/// - In-place partitioning (O(log n) recursion stack)
/// - Faster than sorting for single kth element
///
/// Trade-offs:
/// - Randomized: O(n) expected vs. Median-of-medians: O(n) worst-case
/// - Randomized is simpler and faster in practice
///
/// ### Miller-Rabin Primality Test
/// - `isProbablyPrime()`: Probabilistic primality test - O(k log³ n) time
/// - `isPrime()`: Deterministic test (all 64-bit integers) - O(log³ n) time
/// - `nextPrime()`: Find next prime number - O(n log³ n / ln n) expected
/// - `randomPrime()`: Generate random prime in range - O(k log³ n) time
///
/// Properties:
/// - Monte Carlo algorithm: may have false positives (composite reported as prime)
/// - Error probability ≤ 4^(-k) where k = number of rounds
/// - Recommended k=10 for practical use (error ≤ 10^-6)
/// - Deterministic variant using fixed witnesses for all 64-bit integers
///
/// Applications:
/// - Cryptography: RSA key generation
/// - Hash table sizing: prime-sized tables
/// - Number theory computations
///
/// ## Randomization Benefits
///
/// 1. **Simplicity**: Often simpler than deterministic alternatives
/// 2. **Average-case efficiency**: Expected O(n) vs. worst-case O(n log n)
/// 3. **Adversary resistance**: No worst-case input exists (probabilistically)
/// 4. **Practical performance**: Constant factors usually better
///
/// ## Random Number Sources
///
/// All algorithms accept `std.Random` interface:
/// ```zig
/// var prng = std.Random.DefaultPrng.init(seed);
/// const random = prng.random();
/// ```
///
/// For cryptographic applications, use `std.crypto.random` instead.
///
/// ## Example Usage
///
/// ```zig
/// const std = @import("std");
/// const randomized = @import("zuda").algorithms.randomized;
///
/// var prng = std.Random.DefaultPrng.init(42);
/// const random = prng.random();
///
/// // Shuffle array
/// var cards = [_]u8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
/// randomized.fisher_yates.shuffle(u8, &cards, random);
///
/// // Find median
/// var data = [_]i32{9, 1, 8, 2, 7, 3, 6, 4, 5};
/// const med = randomized.randomized_select.median(
///     i32, &data, random,
///     struct { fn lt(_: void, a: i32, b: i32) bool { return a < b; } }.lt
/// );
///
/// // Test primality
/// const is_prime = randomized.miller_rabin.isProbablyPrime(104729, 10, random);
///
/// // Reservoir sampling from stream
/// const stream: []const i32 = &[_]i32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
/// const sample = try randomized.reservoir_sampling.reservoirSample(
///     i32, allocator, stream, 3, random
/// );
/// defer allocator.free(sample);
/// ```

pub const fisher_yates = @import("randomized/fisher_yates.zig");
pub const reservoir_sampling = @import("randomized/reservoir_sampling.zig");
pub const randomized_select = @import("randomized/randomized_select.zig");
pub const miller_rabin = @import("randomized/miller_rabin.zig");

// Re-export commonly used functions for convenience
pub const shuffle = fisher_yates.shuffle;
pub const partialShuffle = fisher_yates.partialShuffle;
pub const randomPermutation = fisher_yates.randomPermutation;

pub const reservoirSample = reservoir_sampling.reservoirSample;
pub const weightedReservoirSample = reservoir_sampling.weightedReservoirSample;
pub const combineReservoirs = reservoir_sampling.combineReservoirs;

pub const randomizedSelect = randomized_select.randomizedSelect;
pub const median = randomized_select.median;
pub const topK = randomized_select.topK;

pub const isProbablyPrime = miller_rabin.isProbablyPrime;
pub const isPrime = miller_rabin.isPrime;
pub const nextPrime = miller_rabin.nextPrime;
pub const randomPrime = miller_rabin.randomPrime;

test {
    @import("std").testing.refAllDecls(@This());
}
