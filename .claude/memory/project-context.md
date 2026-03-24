# zuda Project Context

## Current Status
- **Version**: 1.23.0 (current)
- **Phase**: v2.0 Track — Phase 10 IN PROGRESS
- **Zig Version**: 0.15.2
- **Last CI Status**: ✅ GREEN (verified 2026-03-24 Session 18)
- **Latest Milestone**: v1.23.0 ✅ — Numerical Methods (Integration, Differentiation, Interpolation) RELEASED (2026-03-24)
- **Current Milestone**: Phase 10 (Numerical Methods) — 3/6 functions complete (trapezoid, simpson, diff/gradient, interp1d)
- **Next Priority**: Remaining Phase 10 functions (quad, romberg, cubic_spline) or release v1.23.0
- **Test Count**: 1730 tests passing (+87 from v1.22.0)
  - Breakdown: 301 linalg + 102 stats descriptive + 602 distributions + 143 hypothesis tests + 129 correlation/regression + 213 signal + 87 numeric (33 integration + 28 differentiation + 26 interpolation) + ndarray + containers + algorithms + internal
  - Skipped: 1 Normal quantile test (Acklam approximation), 1 mannwhitney empty array (NDArray prevents zero-length)
  - Numerical Methods: 3 functions complete (trapezoid, simpson, diff/gradient, interp1d)
- **System Status**: STABLE — all tests passing, ready for release

## Recent Progress (Session 2026-03-24 - Session 18)
**FEATURE MODE:**

### v1.23.0 Release ✅ (2026-03-24)
- ✅ **Phase 10 PARTIAL COMPLETE**: Numerical Methods foundation (integration, differentiation, interpolation)
- ✅ **Pre-flight checks**: All 1730 tests passing, CI green
- ✅ **Version bump**: 1.22.0 → 1.23.0
- ✅ **Tag**: v1.23.0 created and pushed
- ✅ **GitHub Release**: https://github.com/yusa-imit/zuda/releases/tag/v1.23.0
- ✅ **Total tests**: 1730 passing (+87 from v1.22.0)
- ✅ **Modules**: Integration (trapezoid, simpson), Differentiation (diff, gradient), Interpolation (interp1d)

### Numerical Methods Foundation (commits 4430183, 49d872b, c495181) ✅
- ✅ **Modules**: 3 new modules in src/numeric/ (1828 lines total: 235 implementation + 1593 tests)
- ✅ **TDD Workflow**: test-writer (87 tests) → implementations → all tests passing
- ✅ **Integration Module** (src/numeric/integration.zig, 631 lines: 86 impl + 545 tests):
  - **trapezoid(T, x, y, allocator) !T** — Trapezoidal rule integration
    - Exact for polynomials degree ≤ 1 (linear)
    - Formula: ∫f(x)dx ≈ Σ(x[i+1] - x[i]) * (f[i] + f[i+1]) / 2
    - Time: O(n), Space: O(1)
    - Tests: 13 (constant, linear, quadratic, sin/cos/exp, errors, f32/f64)
  - **simpson(T, x, y, allocator) !T** — Simpson's rule integration
    - Exact for polynomials degree ≤ 3 (cubic)
    - Formula: (h/3) * Σ(f[i] + 4*f[i+1] + f[i+2])
    - Validates odd length (Simpson requires odd points)
    - Time: O(n), Space: O(1)
    - Tests: 13 (quadratic/cubic exact, non-uniform grids, edge cases)
  - Comparative tests: 7 (convergence, accuracy comparison)
- ✅ **Differentiation Module** (src/numeric/differentiation.zig, 611 lines: 73 impl + 538 tests):
  - **diff(T, y, dx, allocator) ![]T** — Finite difference differentiation
    - Forward difference at i=0: (y[1] - y[0]) / dx
    - Central difference (interior): (y[i+1] - y[i-1]) / (2*dx)
    - Backward difference at i=n-1: (y[n-1] - y[n-2]) / dx
    - Returns allocated array (caller owns)
    - Time: O(n), Space: O(n)
    - Tests: 23 (constant/linear/quadratic, sin/cos/exp, boundaries, errors)
  - **gradient(T, y, dx, allocator) ![]T** — Alias for diff() (NumPy compatibility)
    - Tests: 5 (API compatibility, types)
- ✅ **Interpolation Module** (src/numeric/interpolation.zig, 577 lines: 76 impl + 501 tests):
  - **interp1d(T, x, y, x_new, allocator) ![]T** — 1D linear interpolation
    - Binary search for interval location
    - Linear interpolation: y_new[i] = y[j] + (y[j+1] - y[j]) * (x_new[i] - x[j]) / (x[j+1] - x[j])
    - Constant extrapolation (clamp to boundary values)
    - Validates x monotonically increasing
    - Time: O(m log n + m), Space: O(m) where m = x_new.len, n = x.len
    - Tests: 26 (exact match, linear exact, extrapolation, non-uniform grids, errors)
- ✅ **Test Count**: 1643 → 1730 (+87 tests: 33 integration + 28 differentiation + 26 interpolation)
- ✅ **Implementation Quality**:
  - Generic over f32/f64 via comptime type parameter
  - Allocator-first design (caller-provided allocator)
  - Big-O complexity documented in all public functions
  - Comprehensive error handling (DimensionMismatch, InsufficientPoints, OddLengthRequired, NonMonotonicX)
  - Memory safety verified with std.testing.allocator
  - Analytical validation (known derivatives/integrals)

---

## Previous Progress (Session 2026-03-24 - Session 17)
**FEATURE MODE:**

### Digital Filter Design & Application (commit abe6f59) ✅
- ✅ **Module**: src/signal/filter.zig (1021 lines: 331 implementation + 690 tests)
- ✅ **TDD Workflow**: test-writer (39 tests) → zig-developer (implementation) → all 39 tests passing
- ✅ **firwin(comptime T, N, cutoff, fs, allocator) ![]T**:
  - FIR filter design using windowed sinc method
  - Hamming window for spectral leakage suppression
  - DC gain normalization (sum of coefficients ≈ 1 for lowpass)
  - Validates cutoff < fs/2 (Nyquist constraint)
  - Time: O(N), Space: O(N)
  - Tests: 8 (design, DC gain, symmetry, type support, errors)
- ✅ **lfilter(comptime T, b, a, x, allocator) ![]T**:
  - Apply IIR/FIR filters via difference equation
  - Direct form II transposed implementation
  - Supports both FIR (a=[1]) and IIR (general a) filters
  - Zero initial conditions for causal filtering
  - Time: O(N·M), Space: O(N)
  - Tests: 14 (FIR/IIR, orders, edge cases, error handling)
- ✅ **filtfilt(comptime T, b, a, x, allocator) ![]T**:
  - Zero-phase filtering via forward-backward pass
  - Eliminates phase distortion for linear-phase applications
  - Mirror padding at boundaries (scipy-compatible)
  - Magnitude response squared: |H(ω)|²
  - Time: O(N·M), Space: O(N)
  - Tests: 8 (zero-phase, symmetry, type support, errors)
- ✅ **butter(comptime T, N, cutoff, fs, allocator) !FilterCoefficients(T)**:
  - Butterworth IIR lowpass filter design
  - Maximally flat passband response
  - Bilinear transformation from analog prototype
  - Explicit implementations for N=1,2
  - All poles guaranteed inside unit circle (stable)
  - Time: O(N²), Space: O(N)
  - Tests: 8 (design, gain, order scaling, type support, errors)
- ✅ **FilterCoefficients(T)** struct:
  - `b: []T` (numerator coefficients)
  - `a: []T` (denominator coefficients)
  - `deinit()` for cleanup
- ✅ **Test Count**: 1604 → 1643 passing (+39 filter tests)
- ✅ **Implementation Quality**:
  - Generic over f32/f64 via comptime type parameter
  - Allocator-first design (no hardcoded allocator)
  - Big-O complexity documented in all public functions
  - Comprehensive error handling (InvalidArgument)
  - Memory safety verified with std.testing.allocator

### v1.22.0 Release ✅ (2026-03-24)
- ✅ **Phase 9 COMPLETE**: Signal Processing module (7 submodules, 213 tests)
- ✅ **Pre-flight checks**: All tests passing, 6 cross-compile targets verified
- ✅ **Version bump**: 1.21.0 → 1.22.0
- ✅ **Tag**: v1.22.0 created and pushed
- ✅ **GitHub Release**: https://github.com/yusa-imit/zuda/releases/tag/v1.22.0
- ✅ **Total tests**: 1643 passing
- ✅ **Modules**: FFT, Window, Spectral, DCT, Convolution, 2D FFT, Filtering

---

## Previous Progress (Session 2026-03-24 - Session 16)
**FEATURE MODE:**

### 2D FFT Implementation (commit cc33699) ✅
- ✅ **Module**: src/signal/fft2d.zig (1002 lines: 232 implementation + 770 tests)
- ✅ **TDD Workflow**: test-writer (22 tests) → zig-developer (implementation) → all 22 tests passing
- ✅ **fft2(comptime T, signal2d: NDArray(T), allocator) !NDArray(Complex(T))**:
  - 2D Fast Fourier Transform for image and 2D signal processing
  - Algorithm: Row-then-column decomposition (separable transform)
    1. Apply 1D FFT to each row
    2. Apply 1D FFT to each column of row-FFT result
  - Input: real or complex 2D array (M×N)
  - Output: complex 2D array (M×N) with full spectrum
  - Validates M and N are powers of 2 (error.InvalidLength)
  - Layout preservation (row_major/column_major)
  - Time: O(MN log(MN)), Space: O(MN)
  - Supports f32 and f64 via comptime type parameter
- ✅ **ifft2(comptime T, spectrum2d: NDArray(Complex(T)), allocator) !NDArray(Complex(T))**:
  - 2D inverse FFT — mathematical inverse of fft2
  - Same row-then-column structure using ifft() instead of fft()
  - Round-trip: ifft2(fft2(x)) ≈ x within floating-point precision
  - Time: O(MN log(MN)), Space: O(MN)
- ✅ **Test Coverage** (22 tests):
  - Basic operations (5): 2×2 impulse, round-trip 2×2/4×4/8×8/16×16
  - Mathematical properties (5): DC component (X[0,0] = sum), all zeros/ones, linearity, energy conservation
  - Edge cases (6): single row (4×1), single column (1×4), non-square (4×8, 8×4), checkerboard, diagonal
  - Type support (3): f32, f64, layout preservation (row/column major)
  - Error handling (2): non-power-of-2 rows/cols validation
  - Memory safety: proper allocation/deallocation verified
- ✅ **File**: src/signal/fft2d.zig (1002 lines) + root.zig update
- ✅ **Test Count**: 1582 → 1604 passing (+22 2D FFT tests)

**Session 12 Previous Progress:**

### DCT (Discrete Cosine Transform) Implementation (commit d8ab664) ✅
- ✅ **Module**: src/signal/dct.zig (596 lines: 97 implementation + 499 tests)
- ✅ **TDD Workflow**: test-writer (30 tests) → zig-developer (implementation + test fix) → all 30 tests passing
- ✅ **dct(comptime T, signal, allocator) ![]T**:
  - DCT Type II (forward transform) for signal compression/frequency analysis
  - Algorithm: Naive O(N²) computation using cosine basis function
  - Formula: X[k] = sum_{n=0}^{N-1}( x[n] * cos(π * k * (n + 0.5) / N) )
  - Orthonormal scaling: sqrt(1/N) for k=0 (DC), sqrt(2/N) for k>0 (AC)
  - Energy conservation: sum(dct(x)[k]²) ≈ sum(x[n]²)
  - Returns allocated slice (caller owns, must free)
  - Time: O(N²), Space: O(N)
  - Supports f32 and f64 via comptime type parameter
- ✅ **idct(comptime T, coeffs, allocator) ![]T**:
  - DCT Type III (inverse transform) — true mathematical inverse of DCT-II
  - Same scaling structure ensures idct(dct(x)) ≈ x within float precision
  - Time: O(N²), Space: O(N)
- ✅ **Test Coverage** (30 tests):
  - Basic operations (5): empty, single element, constant signals, impulse
  - Round-trip verification (6): various sizes, f32/f64, non-power-of-2
  - Mathematical properties (5): energy conservation, DC component, orthogonality, linearity, coefficient decay
  - Edge cases (7): zero, negative, mixed, large/small magnitudes, alternating signal
  - Type support (2): f32 and f64
  - Memory safety (4): allocation/deallocation for dct and idct
  - IDCT specific (2): empty coefficients, single coefficient
- ✅ **Bug Fix**: Removed duplicate defer in "dct followed by idct multiple times" test (was causing double-free)
- ✅ **File**: src/signal/dct.zig (596 lines) + root.zig update (added to signal namespace + explicit import for tests)
- ✅ **Test Count**: 1552 → 1582 passing (+30 DCT tests)

**Session 11 Previous Progress:**

### Spectral Analysis Implementation (commit c8b2f1c) ✅
- ✅ **Module**: src/signal/spectral.zig (163 lines implementation)
- ✅ **TDD Workflow**: test-writer (28 tests) → zig-developer (implementation) → all tests passing
- ✅ **periodogram(T, signal, fs, allocator) !PeriodogramResult(T)**:
  - Single FFT-based power spectral density estimate
  - Algorithm: rfft(signal) → power = |FFT[k]|²/N²
  - Single-sided spectrum: 2x scaling for non-DC/Nyquist bins
  - Returns positive frequencies only (0 to fs/2)
  - Time: O(N log N), Space: O(N)
  - Validates: power-of-2 length, non-empty signal, fs > 0
  - PeriodogramResult: struct { frequencies: []T, power: []T } (caller owns)
  - Tests: 13 (sinusoid detection, DC, Parseval's theorem, white noise, errors, memory)
- ✅ **welch(T, signal, fs, nperseg, noverlap, allocator) !WelchResult(T)**:
  - Welch's method: averaged periodograms with overlapping segments
  - Reduces variance through segment averaging (smoother PSD estimate)
  - Algorithm:
    1. Segment signal with stride = nperseg - noverlap
    2. Apply Hann window to each segment
    3. Compute periodogram of windowed segment
    4. Average power across all K segments
  - Auto-rounds segment length to largest power-of-2 ≤ signal.len
  - Window normalization: scale by sum(w²) to account for attenuation
  - Time: O(K·M log M) where K = segments, M = nperseg
  - WelchResult: struct { frequencies: []T, power: []T }
  - Tests: 15 (variance reduction, segment counting, overlap configs, errors, memory)
- ✅ **Implementation Quality**:
  - Generic over f32/f64 via comptime type parameter
  - Proper power normalization for energy conservation (Parseval's theorem)
  - Single-sided spectrum with conjugate symmetry compensation
  - Memory safety: zero leaks verified with std.testing.allocator
  - Error handling: empty arrays, invalid parameters, dimension validation
- ✅ **File**: src/signal/spectral.zig (1014 lines: 163 implementation + 851 tests)
- ✅ **Use Cases**: Frequency domain analysis, noise characterization, spectral leakage reduction, audio/sensor signal analysis
- ✅ **Test Count**: 1524 → 1552 passing (+28 tests)

**v1.22.0 COMPLETE** ✅ (Released 2026-03-24):
- [x] FFT (fft, ifft, rfft, irfft, fftfreq) ✅
- [x] Window Functions (hamming, hann, blackman, bartlett, kaiser) ✅
- [x] Spectral Analysis (periodogram, welch) ✅
- [x] DCT (dct, idct) ✅
- [x] Convolution (convolve, correlate, fftconvolve) ✅
- [x] 2D FFT (fft2, ifft2) ✅
- [x] Filtering (firwin, lfilter, filtfilt, butter) ✅

**Next Session Priority**: Phase 10 (Numerical Methods) — integration, differentiation, interpolation

---

## Previous Progress (Session 2026-03-24 - Hour 7)
- ✅ **Functions**: periodogram, welch — Power spectral density estimation
**FEATURE MODE:**

### Logistic Regression Implementation (commit 4de9fcf) ✅
- ✅ **Function**: logisticRegress(T, X, y, allocator) !LogisticRegressionResult(T) — Binary logistic regression for classification
- ✅ **Algorithm**: Newton-Raphson optimization with Iterative Reweighted Least Squares (IRLS)
  - Maximum likelihood estimation via iterative optimization
  - Hessian: H = X^T · W · X where W = diag(p·(1-p))
  - Gradient: g = X^T · (p - y)
  - Update: β_new = β_old - H^-1 · g
  - Convergence: ||Δβ|| < 1e-6 or max 100 iterations
  - Time: O(n·d²·iterations), Space: O(n·d)
- ✅ **LogisticRegressionResult(T)** struct:
  - coefficients: []T (n_features) — caller owns, must free
  - intercept: T
  - log_likelihood: T — final log-likelihood value
  - n_iter: usize — number of iterations until convergence
- ✅ **Implementation Quality**:
  - Generic over f32/f64 via comptime type parameter
  - Newton-Raphson with IRLS for fast convergence
  - Gaussian elimination with partial pivoting for solving H·Δβ = -g
  - Proper sigmoid function: σ(z) = 1/(1 + exp(-z))
  - Validates y contains only 0.0 and 1.0
  - Dimension validation: X.shape[0] == y.shape[0]
  - Memory safety: zero leaks verified with std.testing.allocator
- ✅ **Tests**: 22/22 passing (1674 lines of test code)
  - Basic: perfect separation, good fit, single/multiple features, decision boundary, balanced classes
  - Edge: all y=0/1, minimal samples (n=2), single feature two points, imbalanced classes, identical X
  - Statistical: coefficient signs, result structure, log-likelihood monotonic, convergence < 100 iter
  - Precision: f32 (tolerance 1e-4)
  - Scalability: large dataset (50 samples)
  - Error paths: dimension mismatch, invalid y values (not 0/1), invalid input detection
  - Memory: leak detection, multiple calls no cross-contamination
- ✅ **TDD Workflow**: test-writer (22 tests + implementation) → all tests passing
- ✅ **File**: src/stats/correlation.zig (+805 lines: 191 implementation + 614 tests)
- ✅ **Use Cases**: Binary classification, logistic models, odds ratio estimation, medical diagnosis, spam detection
- ✅ **Test Count**: 1445 → 1467 passing (+22 tests)

**v1.22.0 Progress**:
- [x] Hypothesis Testing (7 tests) ✅
- [x] Correlation (pearson, spearman, kendalltau) ✅
- [x] Simple Linear Regression (linregress) ✅
- [x] Polynomial Regression (polyfit, polyval) ✅
- [x] Logistic Regression (logisticRegress) ✅
- [x] Histogram binning (histogram, histogram2d, histogramBinEdges) ✅

**Next Session Priority**: Ridge/Lasso regression or release v1.22.0

---

## Previous Progress (Session 2026-03-24 - Hour 4)
