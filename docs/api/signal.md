# Signal Processing Module API Reference

**Version**: v2.0.0 | **Module**: `src/signal/` | **Language**: Zig 0.15.x

## Module Overview

The Signal Processing module (`zuda.signal`) provides comprehensive digital signal processing capabilities for frequency domain analysis, filtering, convolution, and spectral estimation. It includes:

- **Fast Fourier Transform (FFT)**: 1D and 2D Cooley-Tukey FFT, real FFT (RFFT)
- **Convolution & Correlation**: Time-domain and FFT-based methods
- **Digital Filtering**: FIR and IIR filter design and application
- **Windowing Functions**: Spectral leakage reduction for FFT analysis
- **Spectral Analysis**: Power spectral density estimation via periodogram and Welch's method
- **Discrete Cosine Transform (DCT)**: Real-valued alternative to FFT

All functions follow Zig conventions: explicit error handling, allocator-first design, and support for f32/f64 floating-point types.

### Module Import
```zig
const signal = @import("zuda").signal;

// Or import specific submodules:
const fft_mod = signal.fft;
const conv = signal.conv;
const filter = signal.filter;
```

---

## 1. Fast Fourier Transform (FFT)

**File**: `src/signal/fft.zig`

### Complex Number Type

```zig
pub fn Complex(comptime T: type) type
```

Represents a complex number with real and imaginary components. Provides arithmetic operations and properties.

#### Methods

**`init(re: T, im: T) -> Complex(T)`**
- Creates a complex number from real and imaginary parts
- Time: O(1) | Space: O(1)

**`add(a: Complex(T), b: Complex(T)) -> Complex(T)`**
- Complex addition: a + b = (a.re + b.re) + i(a.im + b.im)
- Time: O(1)

**`sub(a: Complex(T), b: Complex(T)) -> Complex(T)`**
- Complex subtraction: a - b = (a.re - b.re) + i(a.im - b.im)
- Time: O(1)

**`mul(a: Complex(T), b: Complex(T)) -> Complex(T)`**
- Complex multiplication: a * b = (a.re*b.re - a.im*b.im) + i(a.re*b.im + a.im*b.re)
- Time: O(1)

**`conj(a: Complex(T)) -> Complex(T)`**
- Complex conjugate: conj(a) = a.re - i*a.im
- Time: O(1)

**`magnitude(a: Complex(T)) -> T`**
- Absolute value: |a| = sqrt(a.re² + a.im²)
- Time: O(1)

**`magnitude_squared(a: Complex(T)) -> T`**
- Squared magnitude: |a|² = a.re² + a.im²
- Faster than magnitude() when absolute value not needed
- Time: O(1)

**`phase(a: Complex(T)) -> T`**
- Phase angle: arg(a) = atan2(a.im, a.re) in radians
- Returns angle in [-π, π]
- Time: O(1)

**`eql(a: Complex(T), b: Complex(T), epsilon: T) -> bool`**
- Equality test with floating-point tolerance
- Time: O(1)

---

### Forward FFT: fft()

```zig
pub fn fft(comptime T: type, signal: []const Complex(T), allocator: Allocator)
    (Allocator.Error || error{InvalidLength})![]Complex(T)
```

Computes the Fast Fourier Transform of a complex-valued signal using the Cooley-Tukey algorithm (radix-2).

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal**: Time-domain complex signal. Length MUST be a power of 2
- **allocator**: Memory allocator for output

#### Returns
- Frequency-domain complex spectrum (same length as input)
- Caller owns output; must call `allocator.free(spectrum)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Signal length is not a power of 2

#### Complexity
- Time: O(n log n) where n = signal.len
- Space: O(n) for output storage

#### Properties
- **Parseval's Theorem**: `sum(|x[k]|²) = (1/n) * sum(|X[k]|²)` (energy conservation)
- **DC Component**: `X[0] = sum(x[k])` (sum of all input values)
- **Symmetry for real input**: When input is real, `X[n-k] = conj(X[k])`
- **Round-trip**: `ifft(fft(x)) ≈ x` (within floating-point precision)

#### Example: Computing FFT of a complex signal
```zig
const allocator = std.testing.allocator;

// Create a complex signal: [1+0i, 0+0i, 0+0i, 0+0i] (impulse)
var signal = [_]signal.fft.Complex(f64){
    signal.fft.Complex(f64).init(1.0, 0.0),
    signal.fft.Complex(f64).init(0.0, 0.0),
    signal.fft.Complex(f64).init(0.0, 0.0),
    signal.fft.Complex(f64).init(0.0, 0.0),
};

const spectrum = try signal.fft.fft(f64, signal[0..], allocator);
defer allocator.free(spectrum);

// For impulse: all bins have magnitude 1.0
for (spectrum) |bin| {
    const mag = bin.magnitude();
    std.debug.print("Magnitude: {}\n", .{mag});  // Prints ~1.0
}
```

---

### Inverse FFT: ifft()

```zig
pub fn ifft(comptime T: type, spectrum: []const Complex(T), allocator: Allocator)
    (Allocator.Error || error{InvalidLength})![]Complex(T)
```

Computes the Inverse Fast Fourier Transform, converting frequency domain back to time domain.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **spectrum**: Frequency-domain complex spectrum. Length MUST be a power of 2
- **allocator**: Memory allocator for output

#### Returns
- Time-domain complex signal (same length as input)
- Caller owns output; must call `allocator.free(signal)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Spectrum length is not a power of 2

#### Complexity
- Time: O(n log n)
- Space: O(n)

#### Properties
- **Normalization**: Output is scaled by 1/n to normalize energy
- **Inverse property**: ifft(fft(x)) ≈ x
- **Linearity**: ifft(a*X + b*Y) = a*ifft(X) + b*ifft(Y)

#### Example: Round-trip FFT/IFFT
```zig
const allocator = std.testing.allocator;

// Create original signal
var original = [_]signal.fft.Complex(f64){
    signal.fft.Complex(f64).init(1.0, 0.5),
    signal.fft.Complex(f64).init(2.0, -1.0),
    signal.fft.Complex(f64).init(0.5, 0.0),
    signal.fft.Complex(f64).init(3.0, 2.0),
};

// FFT -> IFFT
const spectrum = try signal.fft.fft(f64, original[0..], allocator);
defer allocator.free(spectrum);

const recovered = try signal.fft.ifft(f64, spectrum, allocator);
defer allocator.free(recovered);

// Verify round-trip (within floating-point precision)
for (original, recovered) |orig, recov| {
    assert(approxEqual(orig.re, recov.re, 1e-10));
    assert(approxEqual(orig.im, recov.im, 1e-10));
}
```

---

### Real FFT: rfft()

```zig
pub fn rfft(comptime T: type, signal: []const T, allocator: Allocator)
    (Allocator.Error || error{InvalidLength})![]Complex(T)
```

Computes the Real Fast Fourier Transform, exploiting conjugate symmetry of real-valued signals. Returns only positive frequencies (0 to fs/2), reducing output size by half.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal**: Real-valued time-domain signal. Length MUST be a power of 2
- **allocator**: Memory allocator for output

#### Returns
- Complex spectrum with `n/2 + 1` bins (positive frequencies only)
- Caller owns output; must call `allocator.free(spectrum)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Signal length is not a power of 2

#### Complexity
- Time: O(n log n) where n = signal.len
- Space: O(n) for intermediate complex buffer; O(n/2) for output

#### Properties
- **Conjugate Symmetry**: Full spectrum can be reconstructed as X[n-k] = conj(X[k])
- **Energy Efficiency**: ~50% smaller output than complex FFT for real signals
- **DC/Nyquist**: First bin is DC (average), last bin is Nyquist frequency (when n > 2)
- **Equivalence**: `rfft(x)` matches first n/2+1 bins of `fft(complex_x)` for real x

#### Example: Real FFT of audio signal
```zig
const allocator = std.testing.allocator;

// Real-valued audio signal (4 samples)
const signal = [_]f64{ 1.0, 2.0, 0.5, 3.0 };

// RFFT returns 3 complex values (n/2 + 1)
const spectrum = try signal.fft.rfft(f64, signal[0..], allocator);
defer allocator.free(spectrum);

// spectrum.len == 3
// spectrum[0] = DC component (0 Hz)
// spectrum[1] = 1st positive frequency (0.25 * fs)
// spectrum[2] = Nyquist frequency (0.5 * fs)

std.debug.print("Number of frequency bins: {}\n", .{spectrum.len});  // 3
```

---

### Inverse Real FFT: irfft()

```zig
pub fn irfft(comptime T: type, spectrum: []const Complex(T), allocator: Allocator)
    Allocator.Error![]T
```

Reconstructs a real-valued time-domain signal from real FFT spectrum. Inverse of rfft().

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **spectrum**: Complex spectrum from rfft() with `m+1` bins where output length = 2m
- **allocator**: Memory allocator for output

#### Returns
- Real-valued time-domain signal of length `2*(spectrum.len - 1)`
- Caller owns output; must call `allocator.free(signal)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O(n log n) where n = output length
- Space: O(n)

#### Properties
- **Inverse property**: irfft(rfft(x)) ≈ x
- **Parseval for real signals**: `sum(x[n]²) ≈ (1/n) * sum(2*|X[k]|²)` (with DC/Nyquist adjustment)

#### Example: Round-trip RFFT/IRFFT
```zig
const allocator = std.testing.allocator;

const original = [_]f64{ 1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.0, 1.0 };

// RFFT -> IRFFT
const spectrum = try signal.fft.rfft(f64, original[0..], allocator);
defer allocator.free(spectrum);

const recovered = try signal.fft.irfft(f64, spectrum, allocator);
defer allocator.free(recovered);

// Verify round-trip
for (original, recovered) |orig, recov| {
    assert(approxEqual(orig, recov, 1e-10));
}
```

---

### FFT Frequency Bins: fftfreq()

```zig
pub fn fftfreq(comptime T: type, n: usize, d: T, allocator: Allocator)
    Allocator.Error![]T
```

Computes the frequency values corresponding to FFT bin centers. Useful for labeling frequency axis in spectral plots.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **n**: Number of frequency bins (typically FFT length)
- **d**: Sample spacing = 1/sampling_rate. For fs=1000 Hz, d=0.001
- **allocator**: Memory allocator for output

#### Returns
- Array of frequency values (length n)
- Caller owns output; must call `allocator.free(freqs)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O(n)
- Space: O(n)

#### Properties
- **Positive frequencies**: freqs[0..n/2] are 0, df, 2*df, ..., (n/2)*df
- **Negative frequencies**: freqs[n/2+1..n-1] are wrapped to [-fs/2, -df]
- **Symmetry**: freqs[i] + freqs[n-i] = 0 (except DC)
- **Frequency resolution**: df = 1 / (n*d) = fs / n

#### Example: Frequency axis for spectrum plot
```zig
const allocator = std.testing.allocator;

const n = 8;          // FFT size
const fs = 1000.0;    // Sampling frequency (Hz)
const d = 1.0 / fs;   // Sample spacing

const freqs = try signal.fft.fftfreq(f64, n, d, allocator);
defer allocator.free(freqs);

// freqs = [0, 125, 250, 375, -500, -375, -250, -125]
// Positive: DC, then increasing up to Nyquist
// Negative: negative frequencies (right half of standard plot)

for (freqs, 0..) |f, i| {
    std.debug.print("Bin {}: {} Hz\n", .{i, f});
}
```

---

## 2. 2D Fast Fourier Transform (FFT2D)

**File**: `src/signal/fft2d.zig`

### 2D FFT: fft2()

```zig
pub fn fft2(comptime T: type, signal2d: NDArray(Complex(T), 2), allocator: Allocator)
    anyerror!NDArray(Complex(T), 2)
```

Computes the 2D Fast Fourier Transform via row-then-column decomposition. Applies 1D FFT to each row, then to each column of the result.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal2d**: 2D complex NDArray with shape [M, N] where M, N are powers of 2
- **allocator**: Memory allocator for temporary buffers and output

#### Returns
- 2D complex NDArray with same shape [M, N] (frequency domain)
- Caller owns output; must call `result.deinit()`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Row or column length is not a power of 2
- `error.ZeroDimension` — Either dimension is 0
- `error.DimensionMismatch` — Input doesn't have exactly 2 dimensions

#### Complexity
- Time: O(MN(log M + log N)) = O(MN log(MN))
- Space: O(MN) for output + O(max(M,N)) for row/col buffers

#### Properties
- **Separability**: fft2(X[m,n]) = fft_col(fft_row(X[m,n]))
- **DC component**: X[0,0] = sum of all input values
- **Parseval's theorem (2D)**: `sum(|x[m,n]|²) = (1/(MN)) * sum(|X[k,l]|²)`
- **Linearity**: fft2(a*X + b*Y) = a*fft2(X) + b*fft2(Y)
- **Conjugate symmetry** (for real input): X[M-m, N-n] = conj(X[m,n])

#### Example: 2D FFT of an image-like signal
```zig
const allocator = std.testing.allocator;
const ndarray = @import("zuda").ndarray;
const Complex = signal.fft.Complex;

// Create a 4x4 2D complex array
var signal2d = try ndarray.NDArray(Complex(f64), 2).init(
    allocator,
    &[_]usize{4, 4},
    .row_major
);
defer signal2d.deinit();

// Fill with values
for (0..4) |m| {
    for (0..4) |n| {
        const val = @as(f64, @floatFromInt(m * 4 + n));
        signal2d.set(&[_]isize{@intCast(m), @intCast(n)},
                     Complex(f64).init(val, 0.0));
    }
}

// Compute 2D FFT
var spectrum = try signal.fft2d.fft2(f64, signal2d, allocator);
defer spectrum.deinit();

// Access frequency bin [k, l]
const bin = try spectrum.get(&[_]isize{0, 0});
std.debug.print("DC component: {} + {}i\n", .{bin.re, bin.im});
```

---

### 2D Inverse FFT: ifft2()

```zig
pub fn ifft2(comptime T: type, spectrum2d: NDArray(Complex(T), 2), allocator: Allocator)
    anyerror!NDArray(Complex(T), 2)
```

Computes the 2D Inverse Fast Fourier Transform, converting frequency domain back to time domain.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **spectrum2d**: 2D complex NDArray with shape [M, N] where M, N are powers of 2
- **allocator**: Memory allocator for temporary buffers and output

#### Returns
- 2D complex NDArray with same shape [M, N] (time domain)
- Caller owns output; must call `result.deinit()`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Row or column length is not a power of 2

#### Complexity
- Time: O(MN log(MN))
- Space: O(MN)

#### Properties
- **Inverse property**: ifft2(fft2(x)) ≈ x
- **Normalization**: Scaled by 1/(M*N)
- **Preserves layout**: Output layout (row/column major) matches input

#### Example: Round-trip 2D FFT/IFFT
```zig
var original = try ndarray.NDArray(Complex(f64), 2).init(allocator, &[_]usize{4, 4}, .row_major);
defer original.deinit();

// Fill with values...

var spectrum = try signal.fft2d.fft2(f64, original, allocator);
defer spectrum.deinit();

var recovered = try signal.fft2d.ifft2(f64, spectrum, allocator);
defer recovered.deinit();

// Verify round-trip (within floating-point precision)
```

---

## 3. Convolution and Correlation

**File**: `src/signal/conv.zig`

### Linear Convolution: convolve()

```zig
pub fn convolve(comptime T: type, allocator: Allocator, a: []const T, b: []const T)
    Allocator.Error![]T
```

Computes linear convolution of two real-valued sequences via direct (time-domain) method.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **allocator**: Memory allocator for output
- **a**: First input sequence
- **b**: Second input sequence

#### Returns
- Convolution result with length = `a.len + b.len - 1`
- Caller owns output; must call `allocator.free(result)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O((a.len + b.len)²) direct convolution
- Space: O(a.len + b.len)

#### Properties
- **Commutativity**: convolve(a, b) = convolve(b, a)
- **Associativity**: convolve(convolve(a, b), c) = convolve(a, convolve(b, c))
- **Impulse property**: convolve(x, [1, 0, ...]) returns x zero-padded
- **Output formula**: y[n] = sum_{k=0}^{n} a[k] * b[n-k] (zero-padded)

#### Example: Convolving two sequences
```zig
const allocator = std.testing.allocator;

const a = [_]f64{ 1, 2, 3 };
const b = [_]f64{ 1, 1 };

const result = try signal.conv.convolve(f64, allocator, a[0..], b[0..]);
defer allocator.free(result);

// result = [1, 3, 5, 3]
// Calculation:
//   y[0] = a[0]*b[0] = 1*1 = 1
//   y[1] = a[0]*b[1] + a[1]*b[0] = 1*1 + 2*1 = 3
//   y[2] = a[1]*b[1] + a[2]*b[0] = 2*1 + 3*1 = 5
//   y[3] = a[2]*b[1] = 3*1 = 3
```

---

### Cross-Correlation: correlate()

```zig
pub fn correlate(comptime T: type, allocator: Allocator, a: []const T, b: []const T)
    Allocator.Error![]T
```

Computes cross-correlation of two real-valued sequences via direct method. Measures similarity between sequences as a function of lag.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **allocator**: Memory allocator for output
- **a**: Reference signal
- **b**: Comparison signal

#### Returns
- Cross-correlation result with length = `a.len + b.len - 1`
- Caller owns output; must call `allocator.free(result)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O((a.len + b.len)²)
- Space: O(a.len + b.len)

#### Properties
- **Non-commutative**: correlate(a, b) ≠ correlate(b, a) in general
- **Peak at zero lag**: correlate(x, x)[0] is maximum (signal energy)
- **Autocorrelation**: correlate(x, x) shows periodicity of x
- **Relationship**: correlate(a, b) = convolve(a, reverse(b))
- **Peak location**: Indicates time delay of maximum similarity

#### Example: Autocorrelation to find periodicity
```zig
const allocator = std.testing.allocator;

const signal = [_]f64{ 1, 2, 3, 4, 1, 2, 3, 4 };  // Periodic

const autocorr = try signal.conv.correlate(f64, allocator, signal[0..], signal[0..]);
defer allocator.free(autocorr);

// autocorr[0] is maximum (signal energy)
// autocorr[4] should be large (period = 4), indicating repetition
```

---

### FFT-Based Convolution: fftconvolve()

```zig
pub fn fftconvolve(comptime T: type, allocator: Allocator, a: []const T, b: []const T)
    (Allocator.Error || error{InvalidLength})![]T
```

Computes linear convolution via FFT (fast method for large signals). Uses frequency-domain multiplication.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **allocator**: Memory allocator for intermediate and output
- **a**: First input sequence
- **b**: Second input sequence

#### Returns
- Convolution result with length = `a.len + b.len - 1`
- Same result as convolve() but computed in frequency domain
- Caller owns output; must call `allocator.free(result)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Propagated from FFT (internal constraint)

#### Complexity
- Time: O(n log n) where n = next power of 2 ≥ a.len + b.len - 1
- Space: O(n) for FFT buffers

#### Properties
- **Efficiency**: Faster than direct convolution for large signals (typically > 1000 samples)
- **Equivalence**: fftconvolve(a, b) ≈ convolve(a, b) (within floating-point precision)
- **Zero-padding**: Automatically pads inputs to next power of 2
- **Numerical precision**: Differs from direct method due to FFT rounding accumulation

#### Example: Filtering with FFT convolution
```zig
const allocator = std.testing.allocator;

// Large signal (1024 samples)
const signal = try allocator.alloc(f64, 1024);
defer allocator.free(signal);
// Fill signal...

// FIR filter kernel (64 taps)
const filter = try allocator.alloc(f64, 64);
defer allocator.free(filter);
// Fill filter...

// FFT convolution (faster for large signals)
const output = try signal.conv.fftconvolve(f64, allocator, signal, filter);
defer allocator.free(output);

// output.len = 1024 + 64 - 1 = 1087
```

---

## 4. Digital Filtering

**File**: `src/signal/filter.zig`

### Filter Coefficients Structure

```zig
pub fn FilterCoefficients(comptime T: type) type
```

Represents numerator (b) and denominator (a) coefficients for digital filters.

#### Fields
- **b**: []T — Numerator coefficients (FIR part)
- **a**: []T — Denominator coefficients (IIR part, a[0] must be 1.0)
- **allocator**: Allocator — Associated allocator

#### Methods

**`deinit(self: *FilterCoefficients(T))`**
- Frees both b and a coefficient arrays
- Must be called when done with filter

---

### FIR Filter Design: firwin()

```zig
pub fn firwin(comptime T: type, N: usize, cutoff: T, fs: T, allocator: Allocator)
    (Allocator.Error || error{InvalidArgument})![]T
```

Designs a finite impulse response (FIR) lowpass filter using the windowed sinc method with Hamming window.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **N**: Filter order (filter length = N+1). Typically odd for center symmetry
- **cutoff**: Cutoff frequency (Hz). Must satisfy 0 < cutoff < fs/2
- **fs**: Sampling frequency (Hz)
- **allocator**: Memory allocator for coefficients

#### Returns
- Array of N+1 filter coefficients (numerator only, for FIR)
- Caller owns output; must call `allocator.free(coeffs)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidArgument` — cutoff >= fs/2 (violates Nyquist)

#### Complexity
- Time: O(N)
- Space: O(N)

#### Properties
- **Linear phase**: Symmetric coefficients h[n] = h[N-n]
- **DC gain**: sum(coeffs) ≈ 1 for lowpass filters
- **Windowing**: Hamming window reduces Gibbs phenomenon
- **Passband ripple**: < 1% (Hamming characteristic)
- **Stopband attenuation**: ~-53 dB

#### Example: Designing a 100 Hz lowpass filter at 1000 Hz sampling
```zig
const allocator = std.testing.allocator;

const fs = 1000.0;      // Sampling frequency (Hz)
const cutoff = 100.0;   // Cutoff frequency (Hz)
const N = 50;           // Filter order (51 taps)

const h = try signal.filter.firwin(f64, N, cutoff, fs, allocator);
defer allocator.free(h);

// h contains 51 symmetric lowpass filter coefficients
// Can be applied via lfilter()
```

---

### Linear Filter Application: lfilter()

```zig
pub fn lfilter(comptime T: type, b: []const T, a: []const T, x: []const T, allocator: Allocator)
    (Allocator.Error || error{InvalidArgument})![]T
```

Applies a digital filter (FIR or IIR) using the difference equation and direct form II transposed implementation.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **b**: Numerator coefficients (FIR part)
- **a**: Denominator coefficients (IIR part). a[0] MUST equal 1.0
- **x**: Input signal
- **allocator**: Memory allocator for output

#### Returns
- Filtered output signal (same length as x)
- Caller owns output; must call `allocator.free(y)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidArgument` — Empty b/a, or a[0] ≠ 1.0

#### Complexity
- Time: O(N·M) where N = signal length, M = max(len(b), len(a))
- Space: O(N + M)

#### Properties
- **Difference equation**: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k]) for k ≥ 1
- **Causality**: Uses only past and present samples, no future samples
- **Direct form II transposed**: Numerically stable implementation
- **Initial conditions**: Assumed zero (no pre-filtering history)

#### Example: Simple moving average (FIR filter)
```zig
const allocator = std.testing.allocator;

const b = [_]f64{ 0.5, 0.5 };   // Moving average of 2 samples
const a = [_]f64{ 1.0 };        // Pure FIR (no feedback)
const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

const y = try signal.filter.lfilter(f64, b[0..], a[0..], x[0..], allocator);
defer allocator.free(y);

// y[0] = 0.5*1.0 = 0.5
// y[1] = 0.5*2.0 + 0.5*1.0 = 1.5
// y[2] = 0.5*3.0 + 0.5*2.0 = 2.5
// y[3] = 0.5*4.0 + 0.5*3.0 = 3.5
```

#### Example: First-order IIR lowpass filter
```zig
const allocator = std.testing.allocator;

// Simple exponential smoothing: y[n] = x[n] + 0.9*y[n-1]
const b = [_]f64{ 1.0 };
const a = [_]f64{ 1.0, -0.9 };  // Feedback coefficient
const x = [_]f64{ 1.0, 0.0, 0.0, 0.0 };

const y = try signal.filter.lfilter(f64, b[0..], a[0..], x[0..], allocator);
defer allocator.free(y);

// y[0] = 1.0
// y[1] = 0.0 + 0.9*1.0 = 0.9
// y[2] = 0.0 + 0.9*0.9 = 0.81
// y[3] = 0.0 + 0.9*0.81 = 0.729
```

---

### Zero-Phase Filtering: filtfilt()

```zig
pub fn filtfilt(comptime T: type, b: []const T, a: []const T, x: []const T, allocator: Allocator)
    (Allocator.Error || error{InvalidArgument})![]T
```

Applies zero-phase filtering via forward-backward pass. Eliminates phase distortion from causal filtering while doubling the magnitude response.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **b**: Numerator coefficients
- **a**: Denominator coefficients (a[0] = 1.0)
- **x**: Input signal
- **allocator**: Memory allocator for output

#### Returns
- Zero-phase filtered output (same length as x)
- Caller owns output; must call `allocator.free(y)`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidArgument` — Empty b/a, x is empty, or a[0] ≠ 1.0

#### Complexity
- Time: O(N·M) for each pass (forward + backward)
- Space: O(N) + O(pad_len) for mirror padding

#### Properties
- **Zero phase distortion**: Linear phase equivalent
- **Magnitude response**: |H(w)|² (squared, doubled effect)
- **Symmetric output**: Input symmetry preserved
- **Implementation**: Mirror-pads input, filters forward+backward, extracts center
- **Edge effects**: May be visible at signal boundaries

#### Example: Zero-phase lowpass filtering
```zig
const allocator = std.testing.allocator;

// Design lowpass FIR filter
const b = try signal.filter.firwin(f64, 20, 0.2, 1.0, allocator);
defer allocator.free(b);

const a = [_]f64{ 1.0 };
const x = [_]f64{ 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0 };

// Zero-phase filtering
const y = try signal.filter.filtfilt(f64, b, a[0..], x[0..], allocator);
defer allocator.free(y);

// y has no phase lag
// Good for post-processing (non-real-time)
```

---

### Butterworth IIR Filter Design: butter()

```zig
pub fn butter(comptime T: type, N: usize, cutoff: T, fs: T, allocator: Allocator)
    (Allocator.Error || error{InvalidArgument})!FilterCoefficients(T)
```

Designs a Butterworth lowpass IIR filter with maximally flat passband magnitude response using the bilinear transformation.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **N**: Filter order (number of poles)
- **cutoff**: Cutoff frequency (Hz, -3 dB point). Must satisfy 0 < cutoff < fs/2
- **fs**: Sampling frequency (Hz)
- **allocator**: Memory allocator for coefficients

#### Returns
- FilterCoefficients struct with b and a arrays (b length = N+1, a length = N+1)
- Caller owns both arrays; must call `coeffs.deinit()`

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidArgument` — cutoff >= fs/2

#### Complexity
- Time: O(N²)
- Space: O(N)

#### Properties
- **Analog prototype**: Butterworth poles at s = e^(j·π·(2k+N-1)/(2N))
- **DC gain**: ~1.0 (normalized to 1.0 at f=0)
- **Cutoff**: Magnitude = -3 dB (0.707) at cutoff frequency
- **Passband ripple**: Zero (maximally flat)
- **Rolloff**: ~20N dB/decade in stopband
- **Stability**: All poles inside unit circle (stable)
- **Implemented orders**: 1 and 2 fully optimized; higher orders use cascade approach

#### Example: Second-order Butterworth filter at 200 Hz
```zig
const allocator = std.testing.allocator;

const fs = 1000.0;      // Sampling frequency
const cutoff = 200.0;   // Cutoff frequency (-3 dB point)
const N = 2;            // Second-order (2 poles)

var coeffs = try signal.filter.butter(f64, N, cutoff, fs, allocator);
defer coeffs.deinit();

// Apply filter
const x = [_]f64{ 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0 };
const y = try signal.filter.lfilter(f64, coeffs.b, coeffs.a, x[0..], allocator);
defer allocator.free(y);

// y is lowpass filtered at 200 Hz
```

---

## 5. Windowing Functions

**File**: `src/signal/window.zig`

Window functions reduce spectral leakage in FFT analysis by tapering signal edges to zero.

### Hamming Window: hamming()

```zig
pub fn hamming(comptime T: type, n: usize, allocator: Allocator)
    Allocator.Error![]T
```

Hamming window: w[n] = α - β·cos(2πn/(N-1)) where α = 0.54, β = 0.46

#### Properties
- **Main lobe width**: 8π/N
- **Sidelobe attenuation**: -43 dB
- **Good for**: General-purpose spectral analysis

#### Example
```zig
const allocator = std.testing.allocator;
const n = 64;

const window = try signal.window.hamming(f64, n, allocator);
defer allocator.free(window);

// window[0] ≈ 0.08 (tapers smoothly)
// window[n/2] ≈ 1.0 (peak at center)
// window[n-1] ≈ 0.08 (symmetric)

// Apply to signal for FFT
for (0..n) |i| {
    windowed_signal[i] = signal[i] * window[i];
}
```

---

### Hann Window: hann()

```zig
pub fn hann(comptime T: type, n: usize, allocator: Allocator)
    Allocator.Error![]T
```

Hann (Hanning) window: w[n] = 0.5·(1 - cos(2πn/(N-1)))

#### Properties
- **Main lobe width**: 8π/N
- **Sidelobe attenuation**: -31 dB
- **Zero endpoints**: w[0] = w[n-1] = 0
- **Good for**: Smooth taper, overlap-add processing

---

### Blackman Window: blackman()

```zig
pub fn blackman(comptime T: type, n: usize, allocator: Allocator)
    Allocator.Error![]T
```

Blackman window: w[n] = α₀ - α₁·cos(2πn/(N-1)) + α₂·cos(4πn/(N-1))

#### Properties
- **Main lobe width**: 12π/N
- **Sidelobe attenuation**: -58 dB
- **Good for**: Better sidelobe suppression (narrower main lobe vs Hamming/Hann)

---

### Bartlett Window: bartlett()

```zig
pub fn bartlett(comptime T: type, n: usize, allocator: Allocator)
    Allocator.Error![]T
```

Bartlett (triangular) window: w[n] = 1 - |n - (N-1)/2| / ((N-1)/2)

#### Properties
- **Triangular shape**: Linear rise then fall
- **Zero endpoints**: w[0] = w[n-1] = 0
- **Main lobe width**: 8π/N
- **Good for**: Simple, weak sidelobe suppression

---

### Kaiser Window: kaiser()

```zig
pub fn kaiser(comptime T: type, n: usize, beta: T, allocator: Allocator)
    Allocator.Error![]T
```

Kaiser window with configurable shape parameter beta for trade-off between main lobe width and sidelobe attenuation.

#### Properties
- **Parameter beta**: Controls sidelobe attenuation
  - beta=0: Rectangular (no tapering)
  - beta=8.6: Similar to Hamming
  - beta=10-20: Strong sidelobe suppression

---

## 6. Spectral Analysis

**File**: `src/signal/spectral.zig`

### Periodogram: periodogram()

```zig
pub fn periodogram(comptime T: type, signal: []const T, fs: T, allocator: Allocator)
    (Allocator.Error || error{ InvalidLength, EmptyArray, InvalidParameter })!PeriodogramResult(T)
```

Computes Power Spectral Density (PSD) via simple FFT-based periodogram. Returns positive frequencies only.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal**: Real-valued time-domain signal. Length MUST be power of 2
- **fs**: Sampling frequency (Hz)
- **allocator**: Memory allocator for result

#### Returns
- **PeriodogramResult(T)**:
  - `frequencies`: Array of length n/2+1, spacing = fs/n
  - `power`: Power values at each frequency

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Signal length is not power of 2
- `error.EmptyArray` — Signal is empty
- `error.InvalidParameter` — fs <= 0

#### Complexity
- Time: O(n log n) via FFT
- Space: O(n)

#### Properties
- **Energy conservation**: sum(power) ≈ mean(signal²) (Parseval)
- **Frequency resolution**: df = fs / n
- **Maximum frequency**: fs/2 (Nyquist)
- **Power units**: |FFT|² / n

#### Example: Spectral analysis of a noisy sine wave
```zig
const allocator = std.testing.allocator;

// Create sine wave at 100 Hz sampled at 1000 Hz
const fs = 1000.0;
const freq = 100.0;
const n = 256;

var signal = try allocator.alloc(f64, n);
defer allocator.free(signal);

for (0..n) |i| {
    const t = @as(f64, @floatFromInt(i)) / fs;
    signal[i] = @sin(2.0 * math.pi * freq * t);
}

const result = try signal.spectral.periodogram(f64, signal, fs, allocator);
defer result.deinit(allocator);

// result.frequencies[i] gives frequency in Hz
// result.power[i] gives power at that frequency
// Expect peak near 100 Hz
```

---

### Welch's Method: welch()

```zig
pub fn welch(comptime T: type, signal: []const T, fs: T,
             segment_len: usize, overlap: f64, allocator: Allocator)
    (Allocator.Error || error{ InvalidLength, InvalidParameter })!WelchResult(T)
```

Computes PSD via Welch's method: averages periodograms of overlapping windowed segments for smoother estimates.

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal**: Real-valued time-domain signal
- **fs**: Sampling frequency (Hz)
- **segment_len**: Length of each segment (should be power of 2)
- **overlap**: Overlap fraction ∈ [0, 1). Typical: 0.5 (50% overlap)
- **allocator**: Memory allocator for result

#### Returns
- **WelchResult(T)**: frequencies and smoothed power estimates

#### Errors
- `error.OutOfMemory` — Allocation failure
- `error.InvalidLength` — Segment length not power of 2
- `error.InvalidParameter` — fs <= 0 or overlap out of range

#### Complexity
- Time: O(K·M log M) where K = num_segments, M = segment_len
- Space: O(max(M, n))

#### Properties
- **Smoother than periodogram**: Averaging reduces variance
- **Variance reduction**: ~K-fold (K = number of segments)
- **Trade-off**: Lower variance, lower frequency resolution
- **Windowing**: Usually applies Hamming or Hann window to each segment
- **Efficiency**: Faster than computing periodogram of full signal for long signals

#### Example: Welch's method for noise characterization
```zig
const allocator = std.testing.allocator;

// Long signal (8192 samples)
const fs = 1000.0;
const signal = try allocator.alloc(f64, 8192);
defer allocator.free(signal);
// Fill signal...

// Welch: 512-sample segments with 50% overlap
const result = try signal.spectral.welch(f64, signal, fs, 512, 0.5, allocator);
defer result.deinit(allocator);

// Smoother PSD than periodogram
// ~16 segments averaged together
```

---

## 7. Discrete Cosine Transform (DCT)

**File**: `src/signal/dct.zig`

Real-valued alternative to FFT, useful for compression (JPEG, audio) and energy concentration.

### Forward DCT: dct()

```zig
pub fn dct(comptime T: type, signal: []const T, allocator: Allocator)
    Allocator.Error![]T
```

Computes DCT Type II: X[k] = sum_{n=0}^{N-1} x[n] · cos(π·k·(n+0.5)/N)

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **signal**: Real-valued time-domain signal
- **allocator**: Memory allocator for output

#### Returns
- DCT coefficients (same length as signal)
- Caller owns output; must call `allocator.free(coeffs)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O(N²) (naive implementation; FFT-based could be O(N log N))
- Space: O(N)

#### Properties
- **Energy concentration**: Most energy in low-frequency coefficients
- **Orthogonal**: Preserves energy (Parseval-like)
- **Real-valued**: Both input and output are real
- **Used in**: JPEG, MP3, image/audio compression

#### Example: Compress signal via DCT coefficient truncation
```zig
const allocator = std.testing.allocator;

const signal = [_]f64{ 1.0, 2.0, 1.5, 0.5, -0.5, 0.0, 1.0, 0.5 };

const coeffs = try signal.dct.dct(f64, signal[0..], allocator);
defer allocator.free(coeffs);

// Zero out high-frequency coefficients (energy is in low freq)
for (4..coeffs.len) |i| {
    coeffs[i] = 0.0;
}

// Reconstruct
const recovered = try signal.dct.idct(f64, coeffs, allocator);
defer allocator.free(recovered);

// recovered ≈ signal but compressed (only first 4 coefficients)
```

---

### Inverse DCT: idct()

```zig
pub fn idct(comptime T: type, coeffs: []const T, allocator: Allocator)
    Allocator.Error![]T
```

Computes DCT Type III (inverse): x[n] = 0.5·X[0] + sum_{k=1}^{N-1} X[k]·cos(π·k·(n+0.5)/N)

#### Parameters
- **T**: Floating-point type (f32 or f64)
- **coeffs**: DCT coefficients from dct()
- **allocator**: Memory allocator for output

#### Returns
- Time-domain signal (same length as input)
- Caller owns output; must call `allocator.free(signal)`

#### Errors
- `error.OutOfMemory` — Allocation failure

#### Complexity
- Time: O(N²)
- Space: O(N)

#### Properties
- **True inverse**: idct(dct(x)) ≈ x
- **Orthonormal scaling**: Preserves energy

---

## Common Usage Patterns

### 1. Frequency Domain Analysis

```zig
// Compute FFT for spectrum analysis
const signal = [_]f64{ /* ... */ };
const spectrum = try signal.fft.rfft(f64, signal[0..], allocator);
defer allocator.free(spectrum);

// Compute magnitudes
for (spectrum) |bin| {
    const magnitude = bin.magnitude();
    const phase = bin.phase();
    std.debug.print("Magnitude: {}, Phase: {}\n", .{magnitude, phase});
}
```

### 2. Digital Filtering

```zig
// Design filter + apply
const h = try signal.filter.firwin(f64, 50, 200.0, 1000.0, allocator);
defer allocator.free(h);

const a = [_]f64{ 1.0 };
const filtered = try signal.filter.lfilter(f64, h, a[0..], input, allocator);
defer allocator.free(filtered);
```

### 3. Signal Convolution (Fast)

```zig
// For large signals, use FFT-based convolution
const convolved = try signal.conv.fftconvolve(f64, allocator, a, b);
defer allocator.free(convolved);
```

### 4. Spectral Leakage Reduction

```zig
// Apply window before FFT
const window = try signal.window.hamming(f64, n, allocator);
defer allocator.free(window);

var windowed = try allocator.alloc(f64, n);
defer allocator.free(windowed);

for (0..n) |i| {
    windowed[i] = input[i] * window[i];
}

const spectrum = try signal.fft.rfft(f64, windowed, allocator);
defer allocator.free(spectrum);
```

---

## Performance Characteristics

| Function | Time | Space | Best For |
|----------|------|-------|----------|
| fft() | O(n log n) | O(n) | Complex signals, frequency analysis |
| rfft() | O(n log n) | O(n) | Real signals (50% smaller output) |
| convolve() | O(n²) | O(n) | Small signals (< 100 samples) |
| fftconvolve() | O(n log n) | O(n) | Large signals (> 1000 samples) |
| correlate() | O(n²) | O(n) | Signal matching, autocorrelation |
| lfilter() | O(n·m) | O(n) | Real-time filtering |
| firwin() | O(n) | O(n) | FIR filter design |
| butter() | O(n²) | O(n) | IIR filter design |
| periodogram() | O(n log n) | O(n) | Quick PSD estimate |
| welch() | O(k·n log n) | O(n) | Smooth PSD, long signals |
| dct() | O(n²) | O(n) | Compression, energy concentration |

---

## Error Handling

All signal processing functions follow these conventions:

```zig
// Check return errors
const spectrum = fft(f64, signal, allocator) catch |err| {
    switch (err) {
        error.OutOfMemory => {
            std.debug.print("Memory allocation failed\n", .{});
        },
        error.InvalidLength => {
            std.debug.print("Signal length must be power of 2\n", .{});
        },
    }
    return;
};
defer allocator.free(spectrum);
```

---

## Memory Management

All signal processing functions follow allocator-first design:

```zig
// Always use defer to guarantee cleanup
const spectrum = try fft(f64, signal, allocator);
defer allocator.free(spectrum);

// For structures with deinit():
var coeffs = try butter(f64, 2, 200.0, 1000.0, allocator);
defer coeffs.deinit();

var result = try fft2(f64, signal2d, allocator);
defer result.deinit();
```

---

## Thread Safety

**Signal processing functions are NOT thread-safe**. Do not share allocators or call functions concurrently without synchronization. Each thread should use its own allocator instance.

---

## Floating-Point Precision

All functions support both f32 and f64:

```zig
// f64 (double precision)
const spectrum64 = try fft(f64, signal64, allocator);

// f32 (single precision)
const spectrum32 = try fft(f32, signal32, allocator);
```

For high-precision requirements (long signal chains, high-order filters), use f64. For embedded/performance-critical applications, f32 is faster but trades precision.

---

## See Also

- **NDArray Module** (`zuda.ndarray`): For 2D/nD signal arrays and operations
- **Linear Algebra Module** (`zuda.linalg`): For matrix decompositions in advanced DSP
- **Statistics Module** (`zuda.stats`): For signal statistics and probability
- **DSP Guides** (`docs/guides/scientific_computing_signal_processing.md`): Practical DSP tutorials

---

## References

1. Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series"
2. Oppenheim, A. V., Schafer, R. W., & Buck, J. R. (2010). "Discrete-Time Signal Processing" (3rd ed.)
3. Welch, P.D. (1967). "The use of fast Fourier transform for estimation of power spectra"
4. Parks, T. W., & Burrus, C. S. (1987). "Digital Filter Design"
5. Smith, S. W. (1997). "The Scientist and Engineer's Guide to Digital Signal Processing"
6. Ahmed, N., Natarajan, T., & Rao, K. R. (1974). "Discrete cosine transform"
7. Gonzalez, R. C., & Woods, R. E. (2008). "Digital Image Processing"

