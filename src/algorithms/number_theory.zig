//! Number Theory Algorithms
//!
//! This module provides fundamental number theory algorithms essential for:
//! - Cryptography (RSA, Diffie-Hellman)
//! - Modular arithmetic and computational algebra
//! - Prime number computations
//! - Solving Diophantine equations
//!
//! ## Submodules
//!
//! - **gcd**: Greatest Common Divisor, Extended Euclidean Algorithm, modular inverse, Diophantine equations
//! - **modular**: Modular exponentiation, Chinese Remainder Theorem, Euler's totient function
//! - **prime**: Primality testing, prime factorization, sieve of Eratosthenes, prime generation
//!
//! ## Examples
//!
//! ### GCD and Extended Euclidean Algorithm
//! ```zig
//! const nt = @import("zuda").algorithms.number_theory;
//!
//! // Compute GCD
//! const g = nt.gcd.gcd(u64, 48, 18); // 6
//!
//! // Extended GCD for Bézout's identity
//! const result = nt.gcd.extendedGcd(i64, 35, 15);
//! // result.gcd = 5, and 35*result.x + 15*result.y = 5
//!
//! // Modular inverse
//! const inv = nt.gcd.modInverse(i64, 3, 11); // Some(4) because 3*4 ≡ 1 (mod 11)
//! ```
//!
//! ### Modular Arithmetic
//! ```zig
//! const nt = @import("zuda").algorithms.number_theory;
//!
//! // Fast modular exponentiation
//! const result = nt.modular.modPow(u64, 2, 10, 1000); // 2^10 mod 1000 = 24
//!
//! // Chinese Remainder Theorem
//! const a = [_]i64{2, 3, 2};
//! const m = [_]i64{3, 5, 7};
//! const x = nt.modular.crt(i64, &a, &m); // Solution to system of congruences
//!
//! // Euler's totient function
//! const phi = nt.modular.eulerTotient(u64, 9); // 6
//! ```
//!
//! ### Prime Numbers
//! ```zig
//! const nt = @import("zuda").algorithms.number_theory;
//!
//! // Primality test
//! const is_prime = nt.prime.isPrime(u64, 17); // true
//!
//! // Generate primes with sieve
//! var primes = try nt.prime.sieveOfEratosthenes(allocator, 30);
//! defer primes.deinit();
//!
//! // Prime factorization
//! var factors = try nt.prime.primeFactorization(allocator, 60);
//! defer factors.deinit();
//! // factors = [(2,2), (3,1), (5,1)] for 60 = 2^2 * 3 * 5
//! ```
//!
//! ## Performance Characteristics
//!
//! | Algorithm | Time Complexity | Space Complexity | Use Case |
//! |-----------|----------------|------------------|----------|
//! | GCD | O(log min(a,b)) | O(1) | Basic number theory |
//! | Extended GCD | O(log min(a,b)) | O(1) | Modular inverse, Diophantine |
//! | Modular Exponentiation | O(log exp) | O(1) | Cryptography |
//! | CRT | O(n log max(m)) | O(1) | System of congruences |
//! | Euler's Totient | O(√n) | O(1) | Cryptography, counting |
//! | Primality Test | O(√n) | O(1) | Basic checking |
//! | Sieve of Eratosthenes | O(n log log n) | O(n) | Generate many primes |
//! | Prime Factorization | O(√n) | O(log n) | Factor decomposition |
//!
//! ## Implementation Notes
//!
//! ### Type Requirements
//! - GCD/LCM: Works with unsigned and signed integers
//! - Modular arithmetic: Requires integer types (u64, i64, etc.)
//! - Overflow handling: modMul uses Russian peasant for large types
//!
//! ### Numerical Stability
//! - All modular operations handle overflow safely
//! - Extended GCD uses iterative (not recursive) algorithm
//! - Prime factorization handles up to u64::MAX efficiently
//!
//! ### Algorithm Selection
//! - Small numbers (<10^6): Trial division for primes
//! - Range of primes: Sieve of Eratosthenes
//! - Large single values: Miller-Rabin (from randomized module)
//! - Modular inverse: Extended GCD (always O(log n))

pub const gcd = @import("number_theory/gcd.zig");
pub const modular = @import("number_theory/modular.zig");
pub const prime = @import("number_theory/prime.zig");
