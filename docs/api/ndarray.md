# NDArray API Reference

## Overview

NDArray is the core foundation for scientific computing in zuda. It provides a generalized N-dimensional array structure supporting both row-major (C order) and column-major (Fortran order) memory layouts, with efficient stride-based indexing for zero-copy slicing and views.

### Import

```zig
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const Layout = zuda.ndarray.Layout;
```

### Key Features

- **Compile-time rank**: Array dimension count known at compile time
- **Runtime shape**: Dimensions can vary at runtime
- **Flexible layout**: Choose row-major (C) or column-major (Fortran) memory order
- **Zero-copy operations**: Slicing, transposing, and permuting create views
- **Stride-based indexing**: O(1) element access with flexible memory layouts
- **Type-generic**: Works with any numeric type (f32, f64, i32, i64, etc.)
- **Memory safety**: Comprehensive bounds checking and invariant validation

---

## Types and Enums

### Layout Enum

Specifies memory layout order for N-dimensional arrays.

```zig
pub const Layout = enum {
    row_major,      // C order: last dimension varies fastest
    column_major,   // Fortran order: first dimension varies fastest
};
```

**Memory ordering details**:

- **row_major (C order)**: For a 2D array with shape [rows, cols], elements are stored with columns varying fastest (row by row). Strides: [cols, 1]
- **column_major (Fortran order)**: For a 2D array with shape [rows, cols], elements are stored with rows varying fastest (column by column). Strides: [1, rows]

### NDArray Type

```zig
pub fn NDArray(comptime T: type, comptime ndim: usize) type
```

N-dimensional array with compile-time rank and runtime shape.

**Type Parameters**:
- `T`: Element type (typically f32, f64, i32, i64, etc.)
- `ndim`: Number of dimensions (rank) — compile-time constant

**Fields**:
```zig
shape: [ndim]usize              // Size along each dimension
strides: [ndim]usize            // Byte offsets for traversal
data: []T                       // Contiguous element storage
allocator: Allocator            // Memory allocator
layout: Layout                  // Row-major or column-major
```

### Error Types

```zig
pub const Error = error{
    ZeroDimension,              // Invalid shape (0-sized dimension)
    CapacityExceeded,           // Size overflow or mismatch
    IndexOutOfBounds,           // Access beyond valid range
    InvalidPermutation,         // Invalid axis permutation
    ShapeMismatch,              // Shape incompatibility
    InvalidFormat,              // File format error
    UnsupportedVersion,         // Unsupported file version
    DimensionMismatch,          // Dimension count mismatch
    TypeMismatch,               // Type incompatibility
    UnexpectedEOF,              // Premature end of file
    EmptyArray,                 // Empty array operation
    IncompatibleShapes,         // Operations on incompatible shapes
};
```

---

## Lifecycle Functions

### init()

Initialize an N-dimensional array with given shape and layout.

```zig
pub fn init(
    allocator: Allocator,
    shape: []const usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator for data storage
- `shape`: Array shape with ndim elements
- `layout`: Row-major or column-major

**Returns**: Initialized NDArray with zero-filled data

**Errors**:
- `error.ZeroDimension`: Any dimension is 0
- `error.CapacityExceeded`: Shape product overflows usize
- `error.OutOfMemory`: Allocation fails

**Complexity**:
- Time: O(ndim)
- Space: O(prod(shape))

**Example**:
```zig
const allocator = std.heap.page_allocator;

// Create 2D array (3x4 matrix)
var matrix = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
defer matrix.deinit();

// Create 3D tensor (2x3x4)
var tensor = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .column_major);
defer tensor.deinit();
```

### deinit()

Free all allocated memory.

```zig
pub fn deinit(self: *Self) void
```

**Complexity**:
- Time: O(1)
- Space: O(1)

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 20}, .row_major);
defer arr.deinit();  // Cleanup
```

### validate()

Verify internal invariants of the array.

```zig
pub fn validate(self: *const Self) !void
```

**Checks**:
- All dimensions > 0
- Strides match shape and layout
- Data length equals product of shape
- Data pointer is valid

**Complexity**:
- Time: O(ndim)
- Space: O(1)

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 10}, .row_major);
defer arr.deinit();
try arr.validate();  // Assert consistency
```

---

## Construction Functions

### zeros()

Create an array filled with zeros.

```zig
pub fn zeros(
    allocator: Allocator,
    shape: []const usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `shape`: Array shape (ndim elements)
- `layout`: Row-major or column-major

**Returns**: NDArray with all elements set to 0

**Complexity**:
- Time: O(prod(shape))
- Space: O(prod(shape))

**Example**:
```zig
var zeros_matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{5, 5}, .row_major);
defer zeros_matrix.deinit();
```

### ones()

Create an array filled with ones.

```zig
pub fn ones(
    allocator: Allocator,
    shape: []const usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: NDArray with all elements set to 1

**Complexity**:
- Time: O(prod(shape))
- Space: O(prod(shape))

**Example**:
```zig
var ones_vec = try NDArray(f32, 1).ones(allocator, &[_]usize{100}, .row_major);
defer ones_vec.deinit();
```

### full()

Create an array filled with a specific value.

```zig
pub fn full(
    allocator: Allocator,
    shape: []const usize,
    value: T,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `shape`: Array shape
- `value`: Value for all elements
- `layout`: Row-major or column-major

**Returns**: NDArray with all elements set to value

**Complexity**:
- Time: O(prod(shape))
- Space: O(prod(shape))

**Example**:
```zig
var filled = try NDArray(f64, 2).full(allocator, &[_]usize{3, 3}, 3.14, .row_major);
defer filled.deinit();
```

### empty()

Create an uninitialized array (data not zero-filled).

```zig
pub fn empty(
    allocator: Allocator,
    shape: []const usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Note**: Use `zeros()` for deterministic behavior.

**Complexity**:
- Time: O(ndim)
- Space: O(prod(shape))

**Example**:
```zig
var uninitialized = try NDArray(f64, 1).empty(allocator, &[_]usize{1000}, .row_major);
defer uninitialized.deinit();
```

### arange()

Create a 1D array with evenly spaced values in range [start, stop).

```zig
pub fn arange(
    allocator: Allocator,
    start: T,
    stop: T,
    step: T,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `start`: Starting value (inclusive)
- `stop`: Stopping value (exclusive for positive step)
- `step`: Spacing between values (must not be 0)
- `layout`: Row-major or column-major

**Returns**: 1D NDArray with values [start, start+step, start+2*step, ...]

**Errors**:
- `error.ZeroDimension`: step == 0

**Complexity**:
- Time: O(num_elements)
- Space: O(num_elements)

**Example**:
```zig
// Integers: [0, 1, 2, ..., 9]
var int_range = try NDArray(i32, 1).arange(allocator, 0, 10, 1, .row_major);
defer int_range.deinit();

// Floats: [0.0, 0.5, 1.0, ..., 9.5]
var float_range = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 0.5, .row_major);
defer float_range.deinit();

// Backwards: [10, 8, 6, 4, 2]
var reverse = try NDArray(i32, 1).arange(allocator, 10, 0, -2, .row_major);
defer reverse.deinit();
```

### linspace()

Create a 1D array with num evenly spaced values in range [start, stop].

```zig
pub fn linspace(
    allocator: Allocator,
    start: T,
    stop: T,
    num: usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `start`: Starting value (inclusive)
- `stop`: Stopping value (inclusive)
- `num`: Number of evenly spaced samples (must be > 0)
- `layout`: Row-major or column-major

**Returns**: 1D NDArray with num evenly distributed values from start to stop

**Errors**:
- `error.ZeroDimension`: num == 0

**Complexity**:
- Time: O(num)
- Space: O(num)

**Example**:
```zig
// 10 points from 0.0 to 1.0 (inclusive)
var points = try NDArray(f64, 1).linspace(allocator, 0.0, 1.0, 10, .row_major);
defer points.deinit();
// Result: [0.0, 0.111..., 0.222..., ..., 1.0]
```

### fromSlice()

Create an array from an existing slice (copies data).

```zig
pub fn fromSlice(
    allocator: Allocator,
    shape: []const usize,
    data_slice: []const T,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `shape`: Array shape (prod(shape) must equal data_slice.len)
- `data_slice`: Slice containing data elements
- `layout`: Row-major or column-major

**Returns**: NDArray copying data from slice

**Errors**:
- `error.CapacityExceeded`: data_slice.len != prod(shape)

**Complexity**:
- Time: O(prod(shape))
- Space: O(prod(shape))

**Example**:
```zig
const data = [_]f64{1, 2, 3, 4, 5, 6};
var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &data, .row_major);
defer arr.deinit();
```

### fromOwnedSlice()

Create an array from an owned (allocated) slice without copying.

```zig
pub fn fromOwnedSlice(
    allocator: Allocator,
    shape: []const usize,
    owned_data: []T,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator (used for deinit)
- `shape`: Array shape (prod(shape) must equal owned_data.len)
- `owned_data`: Slice of allocated data to take ownership of
- `layout`: Row-major or column-major

**Returns**: NDArray with direct reference to owned_data (no copy)

**Ownership**: Caller must NOT free owned_data after this call — deinit() will handle it

**Errors**:
- `error.CapacityExceeded`: owned_data.len != prod(shape)
- `error.ZeroDimension`: Any dimension is 0

**Complexity**:
- Time: O(ndim)
- Space: O(ndim) for metadata

**Example**:
```zig
const owned = try allocator.alloc(f64, 12);
// Fill owned...
var arr = try NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{3, 4}, owned, .row_major);
defer arr.deinit();  // Frees owned internally
```

### eye()

Create an identity/unit matrix with optional diagonal offset.

```zig
pub fn eye(
    allocator: Allocator,
    rows: usize,
    cols: usize,
    k: isize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `allocator`: Memory allocator
- `rows`: Number of rows
- `cols`: Number of columns
- `k`: Diagonal offset (0 = main, positive = above, negative = below)
- `layout`: Row-major or column-major

**Returns**: 2D NDArray with 1s on the k-th diagonal and 0s elsewhere

**Complexity**:
- Time: O(rows * cols)
- Space: O(rows * cols)

**Example**:
```zig
// 5x5 identity matrix
var identity = try NDArray(f64, 2).eye(allocator, 5, 5, 0, .row_major);
defer identity.deinit();

// 5x5 matrix with 1s on first super-diagonal
var super_diag = try NDArray(f64, 2).eye(allocator, 5, 5, 1, .row_major);
defer super_diag.deinit();

// 4x5 matrix with 1s on second sub-diagonal
var sub_diag = try NDArray(f64, 2).eye(allocator, 4, 5, -2, .row_major);
defer sub_diag.deinit();
```

### identity()

Create an identity matrix (main diagonal only).

```zig
pub fn identity(
    allocator: Allocator,
    rows: usize,
    cols: usize,
    layout: Layout,
) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: 2D NDArray with 1s on main diagonal (equivalent to eye(rows, cols, 0, layout))

**Complexity**:
- Time: O(rows * cols)
- Space: O(rows * cols)

**Example**:
```zig
var I = try NDArray(f64, 2).identity(allocator, 10, 10, .row_major);
defer I.deinit();
```

---

## Shape Manipulation

### reshape()

Reshape the array to a new shape without modifying data order.

```zig
pub fn reshape(
    self: *const Self,
    new_shape: []const usize,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `new_shape`: New shape (prod(new_shape) must equal prod(shape))

**Returns**: Reshaped NDArray with new shape and recalculated strides

**Errors**:
- `error.ZeroDimension`: Any element in new_shape is 0
- `error.CapacityExceeded`: prod(new_shape) != prod(shape)

**Complexity**:
- Time: O(n) where n = prod(shape)
- Space: O(prod(shape)) for new allocation

**Example**:
```zig
var arr = try NDArray(f64, 1).arange(allocator, 0.0, 12.0, 1.0, .row_major);
defer arr.deinit();

// Reshape to 3x4
var reshaped = try arr.reshape(&[_]usize{3, 4});
defer reshaped.deinit();
```

### transpose()

Transpose the array by reversing all axes (zero-copy view).

```zig
pub fn transpose(self: *const Self) Self
```

**Returns**: New NDArray view with reversed shape and strides, same data pointer

**Note**: Modifications to the transposed view affect the original array (shared data)

**Complexity**:
- Time: O(ndim)
- Space: O(1) - view only

**Example**:
```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
defer matrix.deinit();

// Transpose (3x4 → 4x3)
var transposed = matrix.transpose();
// No defer needed for transposed (view doesn't own data)
```

### flatten()

Flatten to 1D, converting to row-major order (zero-copy if contiguous).

```zig
pub fn flatten(
    self: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(T, 1)
```

**Returns**: NDArray(T, 1) with all elements in row-major order

**Complexity**:
- Time: O(1) if contiguous, O(n) if non-contiguous
- Space: O(1) if contiguous, O(n) if non-contiguous

**Example**:
```zig
var tensor = try NDArray(f64, 3).zeros(allocator, &[_]usize{2, 3, 4}, .row_major);
defer tensor.deinit();

var flat = try tensor.flatten();
defer flat.deinit();
// flat.shape = [24]
```

### ravel()

Flatten to 1D with always allocating new memory.

```zig
pub fn ravel(
    self: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(T, 1)
```

**Returns**: New 1D NDArray independent copy

**Complexity**:
- Time: O(n) where n = prod(shape)
- Space: O(n)

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 10}, .row_major);
defer arr.deinit();

var raveled = try arr.ravel();
defer raveled.deinit();
// Always creates new allocation
```

### permute()

Reorder dimensions according to axis permutation.

```zig
pub fn permute(
    self: *const Self,
    axes: []const usize,
) Error!Self
```

**Parameters**:
- `axes`: Array of ndim elements, must be valid permutation of [0..ndim)

**Returns**: New NDArray view with reordered shape and strides

**Errors**:
- `error.InvalidPermutation`: Invalid axes (wrong length, out of range, duplicates)

**Complexity**:
- Time: O(ndim)
- Space: O(1) - view only

**Example**:
```zig
// 3D tensor [2, 3, 4]
var tensor = try NDArray(f64, 3).zeros(allocator, &[_]usize{2, 3, 4}, .row_major);
defer tensor.deinit();

// Permute to [4, 3, 2]
var permuted = try tensor.permute(&[_]usize{2, 1, 0});
```

### contiguous()

Ensure the array has contiguous memory layout.

```zig
pub fn contiguous(
    self: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: New NDArray with contiguous memory (allocates copy)

**Complexity**:
- Time: O(n) where n = prod(shape)
- Space: O(n)

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 10}, .row_major);
defer arr.deinit();

// Get contiguous version
var contig = try arr.contiguous();
defer contig.deinit();
```

---

## Indexing and Slicing

### get()

Get element at multi-dimensional indices with negative indexing.

```zig
pub fn get(
    self: *const Self,
    indices: []const isize,
) Error!T
```

**Parameters**:
- `indices`: Array of ndim signed indices (negative = relative to end)

**Returns**: Element value at computed location

**Errors**:
- `error.IndexOutOfBounds`: Any index out of valid range

**Complexity**:
- Time: O(ndim)
- Space: O(1)

**Example**:
```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
defer matrix.deinit();

// Get element at [1, 2]
const val = try matrix.get(&[_]isize{1, 2});

// Negative indexing: last element = [-1, -1]
const last = try matrix.get(&[_]isize{-1, -1});
```

### set()

Set element at multi-dimensional indices with negative indexing.

```zig
pub fn set(
    self: *Self,
    indices: []const isize,
    value: T,
) void
```

**Parameters**:
- `indices`: Array of ndim signed indices (negative = relative to end)
- `value`: Value to set

**Complexity**:
- Time: O(ndim)
- Space: O(1)

**Example**:
```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
defer matrix.deinit();

// Set element at [0, 0]
matrix.set(&[_]isize{0, 0}, 1.0);

// Negative indexing: last element
matrix.set(&[_]isize{-1, -1}, 99.0);
```

### at()

Get element at flat index with negative indexing.

```zig
pub fn at(
    self: *const Self,
    index: isize,
) Error!T
```

**Parameters**:
- `index`: Flat signed index (negative = relative to end)

**Returns**: Element value at flat index location

**Errors**:
- `error.IndexOutOfBounds`: Index out of valid range

**Complexity**:
- Time: O(1)
- Space: O(1)

**Example**:
```zig
var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 1.0, .row_major);
defer arr.deinit();

// Get first element
const first = try arr.at(0);

// Get last element
const last = try arr.at(-1);
```

### slice()

Create a non-owning view of a sub-region.

```zig
pub fn slice(
    self: *const Self,
    ranges: []const [2]?isize,
) Self
```

**Parameters**:
- `ranges`: Array of [start, stop] pairs for each dimension
  - null start → 0 (beginning)
  - null stop → shape[i] (end)
  - Negative indices count from end
  - Out-of-bounds ranges are clamped

**Returns**: New NDArray view sharing underlying data

**Complexity**:
- Time: O(ndim)
- Space: O(1) - view only

**Example**:
```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{5, 5}, .row_major);
defer matrix.deinit();

// Slice [1:3, 2:4]
var submatrix = matrix.slice(&[_][2]?isize{
    [2]?isize{1, 3},    // rows 1-2
    [2]?isize{2, 4},    // cols 2-3
});

// Slice last row
var last_row = matrix.slice(&[_][2]?isize{
    [2]?isize{-1, null},  // last row
    [2]?isize{null, null},  // all columns
});
```

---

## Element-wise Operations

### add()

Element-wise addition (same shape).

```zig
pub fn add(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

**Errors**:
- `error.ShapeMismatch`: shapes don't match

**Complexity**:
- Time: O(n) where n = prod(shape)
- Space: O(n)

**Example**:
```zig
var a = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 3}, .row_major);
defer a.deinit();
var b = try NDArray(f64, 2).full(allocator, &[_]usize{3, 3}, 2.0, .row_major);
defer b.deinit();

var c = try a.add(&b);
defer c.deinit();
```

### sub()

Element-wise subtraction.

```zig
pub fn sub(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

**Errors**:
- `error.ShapeMismatch`: shapes don't match

**Complexity**:
- Time: O(n)
- Space: O(n)

### mul()

Element-wise multiplication.

```zig
pub fn mul(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

### div()

Element-wise division.

```zig
pub fn div(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

### mod()

Element-wise modulo (integer arrays only).

```zig
pub fn mod(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

---

## Unary Mathematical Functions

### neg()

Element-wise negation.

```zig
pub fn neg(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: New array with all elements negated

**Complexity**:
- Time: O(n)
- Space: O(n)

### abs()

Element-wise absolute value.

```zig
pub fn abs(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### exp()

Element-wise exponential (e^x).

```zig
pub fn exp(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### log()

Element-wise natural logarithm (ln x).

```zig
pub fn log(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### log2()

Element-wise logarithm base 2.

```zig
pub fn log2(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### log10()

Element-wise logarithm base 10.

```zig
pub fn log10(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### sqrt()

Element-wise square root.

```zig
pub fn sqrt(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### pow()

Element-wise power (x^exponent).

```zig
pub fn pow(
    self: *const Self,
    exponent: T,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `exponent`: Power to raise each element to

**Example**:
```zig
var x = try NDArray(f64, 1).arange(allocator, 1.0, 4.0, 1.0, .row_major);
defer x.deinit();

var squared = try x.pow(2.0);
defer squared.deinit();
```

### sin()

Element-wise sine (radians).

```zig
pub fn sin(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### cos()

Element-wise cosine (radians).

```zig
pub fn cos(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### tan()

Element-wise tangent (radians).

```zig
pub fn tan(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### asin()

Element-wise arcsine (radians).

```zig
pub fn asin(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### acos()

Element-wise arccosine (radians).

```zig
pub fn acos(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### atan()

Element-wise arctangent (radians).

```zig
pub fn atan(self: *const Self) (Error || std.mem.Allocator.Error)!Self
```

### atan2()

Element-wise two-argument arctangent.

```zig
pub fn atan2(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!Self
```

**Parameters**:
- `other`: Denominator array (must have same shape)

**Returns**: atan(self[i] / other[i]) element-wise

---

## Comparison Operations

### eq()

Element-wise equality comparison.

```zig
pub fn eq(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

**Returns**: NDArray(bool, ndim) with true where elements equal

**Complexity**:
- Time: O(n)
- Space: O(n)

### ne()

Element-wise inequality comparison.

```zig
pub fn ne(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

### lt()

Element-wise less-than comparison.

```zig
pub fn lt(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

### le()

Element-wise less-than-or-equal comparison.

```zig
pub fn le(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

### gt()

Element-wise greater-than comparison.

```zig
pub fn gt(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

### ge()

Element-wise greater-than-or-equal comparison.

```zig
pub fn ge(
    self: *const Self,
    other: *const Self,
) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim)
```

**Example**:
```zig
var a = try NDArray(f64, 1).arange(allocator, 0.0, 5.0, 1.0, .row_major);
defer a.deinit();
var b = try NDArray(f64, 1).full(allocator, &[_]usize{5}, 2.5, .row_major);
defer b.deinit();

var mask = try a.lt(&b);
defer mask.deinit();
// mask = [true, true, true, false, false]
```

---

## Reduction Operations

### count()

Get number of elements in the array.

```zig
pub fn count(self: *const Self) usize
```

**Returns**: Product of all dimensions

**Complexity**:
- Time: O(ndim)
- Space: O(1)

### isEmpty()

Check if any dimension is zero.

```zig
pub fn isEmpty(self: *const Self) bool
```

**Complexity**:
- Time: O(ndim)
- Space: O(1)

### rank()

Get the number of dimensions (rank).

```zig
pub fn rank(self: *const Self) usize
```

**Returns**: ndim (comptime known)

**Complexity**:
- Time: O(1)
- Space: O(1)

### sum()

Sum all elements in the array.

```zig
pub fn sum(self: *const Self) T
```

**Returns**: Sum of all elements as type T

**Complexity**:
- Time: O(n)
- Space: O(1)

**Example**:
```zig
var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
defer arr.deinit();

const total = arr.sum();  // 12.0
```

### prod()

Product of all elements in the array.

```zig
pub fn prod(self: *const Self) T
```

**Returns**: Product of all elements as type T

**Complexity**:
- Time: O(n)
- Space: O(1)

### mean()

Mean (average) of all elements.

```zig
pub fn mean(self: *const Self) f64
```

**Returns**: Mean as f64 (always floating-point)

**Complexity**:
- Time: O(n)
- Space: O(1)

**Example**:
```zig
var arr = try NDArray(f64, 1).arange(allocator, 1.0, 6.0, 1.0, .row_major);
defer arr.deinit();

const avg = arr.mean();  // 3.0
```

### min()

Minimum element in the array.

```zig
pub fn min(self: *const Self) T
```

**Returns**: Minimum element value as type T

**Complexity**:
- Time: O(n)
- Space: O(1)

### max()

Maximum element in the array.

```zig
pub fn max(self: *const Self) T
```

**Returns**: Maximum element value as type T

**Complexity**:
- Time: O(n)
- Space: O(1)

### argmin()

Index of the minimum element (flat index).

```zig
pub fn argmin(self: *const Self) Error!usize
```

**Returns**: Linear index of first minimum occurrence

**Errors**:
- `error.ZeroDimension`: Array is empty

**Complexity**:
- Time: O(n)
- Space: O(1)

### argmax()

Index of the maximum element (flat index).

```zig
pub fn argmax(self: *const Self) Error!usize
```

**Returns**: Linear index of first maximum occurrence

**Complexity**:
- Time: O(n)
- Space: O(1)

### cumsum()

Cumulative sum of elements.

```zig
pub fn cumsum(
    self: *const Self,
    allocator: Allocator,
) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: New NDArray with same shape containing cumulative sums

**Complexity**:
- Time: O(n)
- Space: O(n)

**Example**:
```zig
var arr = try NDArray(f64, 1).full(allocator, &[_]usize{5}, 1.0, .row_major);
defer arr.deinit();

var cs = try arr.cumsum(allocator);
defer cs.deinit();
// cs = [1.0, 2.0, 3.0, 4.0, 5.0]
```

### cumprod()

Cumulative product of elements.

```zig
pub fn cumprod(
    self: *const Self,
    allocator: Allocator,
) (Error || std.mem.Allocator.Error)!Self
```

**Returns**: New NDArray with same shape containing cumulative products

**Complexity**:
- Time: O(n)
- Space: O(n)

---

## Boolean Reduction Operations

### all()

Check if all elements are true (non-zero).

```zig
pub fn all(self: *const Self) bool
```

**Returns**: true if all elements are non-zero

**Complexity**:
- Time: O(n)
- Space: O(1)

### any()

Check if any element is true (non-zero).

```zig
pub fn any(self: *const Self) bool
```

**Returns**: true if at least one element is non-zero

**Complexity**:
- Time: O(n)
- Space: O(1)

---

## Iteration

### iterator()

Create an iterator for traversing all elements.

```zig
pub fn iterator(self: *const Self) Iterator
```

**Returns**: Iterator struct that supports next() -> ?T

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
defer arr.deinit();

var iter = arr.iterator();
while (iter.next()) |val| {
    // Process element
}
```

---

## Serialization

### save()

Save array to binary file.

```zig
pub fn save(self: *const Self, path: []const u8) !void
```

**Parameters**:
- `path`: File path to save to

**File format**:
1. Magic: 'NDARY' (5 bytes)
2. Version: u32 little-endian
3. Type info: type identifier
4. ndim: u32
5. Shape: ndim × u64 (little-endian)
6. Strides: ndim × u64 (little-endian)
7. Data: raw element bytes

**Example**:
```zig
var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{100, 100}, .row_major);
defer arr.deinit();

try arr.save("/tmp/array.nda");
```

### load()

Load array from binary file.

```zig
pub fn load(
    allocator: Allocator,
    path: []const u8,
) !Self
```

**Parameters**:
- `allocator`: Memory allocator for data
- `path`: File path to load from

**Returns**: Loaded NDArray

**Errors**:
- `error.InvalidFormat`: Wrong magic number or format
- `error.UnsupportedVersion`: Unsupported file version
- `error.TypeMismatch`: Type doesn't match file data
- `error.UnexpectedEOF`: Incomplete file
- File I/O errors

**Example**:
```zig
var loaded = try NDArray(f64, 2).load(allocator, "/tmp/array.nda");
defer loaded.deinit();
```

---

## Common Usage Patterns

### Creating and Initializing Arrays

```zig
const std = @import("std");
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;
const Layout = zuda.ndarray.Layout;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Different ways to create arrays
var zeros = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 10}, .row_major);
defer zeros.deinit();

var ones = try NDArray(f64, 2).ones(allocator, &[_]usize{5, 5}, .row_major);
defer ones.deinit();

var range = try NDArray(f64, 1).arange(allocator, 0.0, 100.0, 1.0, .row_major);
defer range.deinit();

var linsp = try NDArray(f64, 1).linspace(allocator, -1.0, 1.0, 100, .row_major);
defer linsp.deinit();
```

### Element Access

```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
defer matrix.deinit();

// Set and get
matrix.set(&[_]isize{0, 0}, 1.0);
matrix.set(&[_]isize{1, 1}, 2.0);
matrix.set(&[_]isize{2, 3}, 3.0);

const val = try matrix.get(&[_]isize{1, 1});
```

### Shape Operations

```zig
var arr = try NDArray(f64, 1).arange(allocator, 0.0, 12.0, 1.0, .row_major);
defer arr.deinit();

// Reshape
var reshaped = try arr.reshape(&[_]usize{3, 4});
defer reshaped.deinit();

// Transpose
var t = reshaped.transpose();

// Flatten
var flat = try t.flatten();
defer flat.deinit();
```

### Element-wise Operations

```zig
var a = try NDArray(f64, 2).ones(allocator, &[_]usize{5, 5}, .row_major);
defer a.deinit();
var b = try NDArray(f64, 2).full(allocator, &[_]usize{5, 5}, 2.0, .row_major);
defer b.deinit();

var sum = try a.add(&b);
defer sum.deinit();

var product = try a.mul(&b);
defer product.deinit();

var exponential = try a.exp();
defer exponential.deinit();
```

### Reductions and Statistics

```zig
var data = try NDArray(f64, 1).arange(allocator, 1.0, 101.0, 1.0, .row_major);
defer data.deinit();

const total = data.sum();
const average = data.mean();
const minimum = data.min();
const maximum = data.max();
const min_idx = try data.argmin();
const max_idx = try data.argmax();
```

### Slicing and Views

```zig
var matrix = try NDArray(f64, 2).zeros(allocator, &[_]usize{10, 10}, .row_major);
defer matrix.deinit();

// Get first 5 rows and columns
var submatrix = matrix.slice(&[_][2]?isize{
    [2]?isize{0, 5},
    [2]?isize{0, 5},
});

// Get last row
var last_row = matrix.slice(&[_][2]?isize{
    [2]?isize{-1, null},
    [2]?isize{null, null},
});
```

---

## Performance Considerations

1. **Zero-copy operations**: `transpose()`, `permute()`, `slice()`, `iterator()` don't copy data
2. **View safety**: Modifications to views affect original arrays (shared data)
3. **Contiguous layout**: Use `contiguous()` if performance-critical operation requires specific memory pattern
4. **Iterator overhead**: Use `iterator()` for stride-safe traversal over potentially non-contiguous data
5. **Shape updates**: `reshape()`, `permute()` may require allocation if resulting in non-contiguous layout
6. **Type conversions**: Mathematical functions preserve element type (e.g., exp() on i32 returns i32, truncated)

---

## Memory Safety

- All arrays allocate with the provided allocator
- `deinit()` must be called to free memory
- Use `defer arr.deinit()` to ensure cleanup
- Views do NOT own data (don't call deinit on views or risk double-free)
- Slices create views with adjusted data pointers, same underlying allocation
- Transposed/permuted arrays return views, not new allocations
- `validate()` asserts internal consistency for debugging

---

## Thread Safety

NDArray is **not thread-safe**. Access from multiple threads requires external synchronization.

---

## Compatibility

NDArray follows NumPy-inspired API patterns for familiar usage by scientific Python users migrating to Zig. Key differences:

- Type must be specified at compile time (T, ndim)
- Shape is runtime-known but rank is compile-time
- No implicit type conversion (f32 and f64 are distinct types)
- Explicit allocator required for all allocating operations
- Operations return error unions, no exceptions

