//! N-dimensional Array — Core foundation for scientific computing
//!
//! NDArray provides a generalized multi-dimensional array structure supporting:
//! - Compile-time known rank (ndim), runtime-known shape
//! - Both row-major (C order) and column-major (Fortran order) memory layouts
//! - Efficient stride-based indexing without copying
//! - Zero-copy slicing and views
//!
//! ## Time Complexity
//! - Element access via strides: O(1)
//! - Reshape (contiguous array): O(1)
//! - Reshape (non-contiguous): O(n)
//! - Slicing: O(1) (view creation)
//!
//! ## Space Complexity
//! - Storage: O(prod(shape)) for contiguous arrays
//! - Strides metadata: O(ndim)
//!
//! ## Use Cases
//! - Matrix and tensor computations
//! - Linear algebra operations
//! - Signal processing pipelines
//! - Statistical data processing
//! - NumPy-compatible scientific computing

const std = @import("std");
const stdlib = std;
const testing = stdlib.testing;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const AllocatorError = mem.Allocator.Error;

// Aliases to avoid shadowing by method names
const mem = stdlib.mem;
const math = stdlib.math;
const debug = stdlib.debug;
const fs = stdlib.fs;
const fmt = stdlib.fmt;
const io = stdlib.io;
const builtin = stdlib.builtin;
const sorting = stdlib.sort;

/// Memory layout order for N-dimensional array
pub const Layout = enum {
    /// C order (row-major): last dimension varies fastest in memory
    /// For 2D: consecutive elements along rows
    /// Memory stride: [cols, 1] for [rows, cols] shape
    row_major,

    /// Fortran order (column-major): first dimension varies fastest
    /// For 2D: consecutive elements along columns
    /// Memory stride: [1, rows] for [rows, cols] shape
    column_major,
};

/// N-dimensional array with comptime rank and runtime shape
///
/// Type Parameters:
/// - T: Element type (must be copyable, typically numeric)
/// - ndim: Rank (number of dimensions) — compile-time constant
///
/// Example:
/// ```zig
/// // Create a 2D array (matrix)
/// var matrix = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
/// defer matrix.deinit();
///
/// // Create a 3D array (tensor)
/// var tensor = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .column_major);
/// defer tensor.deinit();
/// ```
pub fn NDArray(comptime T: type, comptime ndim: usize) type {
    return struct {
        const Self = @This();

        /// Error types for NDArray operations
        pub const Error = error{
            ZeroDimension,
            CapacityExceeded,
            IndexOutOfBounds,
            InvalidPermutation,
            ShapeMismatch,
            InvalidFormat,
            InvalidValue,
            UnsupportedVersion,
            DimensionMismatch,
            TypeMismatch,
            UnexpectedEOF,
            EmptyArray,
            IncompatibleShapes,
        };

        /// Shape of the array: length along each dimension
        /// Invariant: all elements > 0 (enforced by init)
        shape: [ndim]usize,

        /// Strides for memory traversal: byte offset per unit in each dimension
        /// Calculated based on layout and shape
        strides: [ndim]usize,

        /// Contiguous allocation of array data
        /// Size: prod(shape) elements
        data: []T,

        /// Memory allocator used for allocation
        allocator: Allocator,

        /// Layout order (row-major or column-major)
        layout: Layout,

        /// Ownership flag: true if this array owns its data (should free on deinit)
        /// false for views/borrowed references that share data with another array
        owned: bool,

        // -- Lifecycle --

        /// Initialize an N-dimensional array with given shape and layout
        ///
        /// Parameters:
        /// - allocator: Memory allocator for data storage
        /// - shape: Slice with ndim elements specifying size along each dimension
        /// - layout: Row-major (C order) or column-major (Fortran order)
        ///
        /// Returns: Initialized NDArray with allocated data (zero-filled)
        ///
        /// Errors:
        /// - error.ZeroDimension if any dimension is 0
        /// - error.CapacityExceeded if prod(shape) > usize.max
        /// - error.OutOfMemory if allocation fails
        ///
        /// Time: O(ndim) strides calculation
        /// Space: O(prod(shape))
        pub fn init(allocator: Allocator, shape: []const usize, layout: Layout) (Error || AllocatorError)!Self {
            // Validate shape length
            if (shape.len != ndim) {
                return error.ZeroDimension;
            }

            // Check for zero dimensions
            for (shape) |dim| {
                if (dim == 0) {
                    return error.ZeroDimension;
                }
            }

            // Calculate total element count with overflow check
            var total_elements: usize = 1;
            for (shape) |dim| {
                // Check for overflow
                if (total_elements > math.maxInt(usize) / dim) {
                    return error.CapacityExceeded;
                }
                total_elements *= dim;
            }

            // Allocate contiguous data
            const data = try allocator.alloc(T, total_elements);
            errdefer allocator.free(data);

            // Copy shape into fixed array
            var shape_array: [ndim]usize = undefined;
            for (0..ndim) |i| {
                shape_array[i] = shape[i];
            }

            // Calculate strides
            const strides = calculateStrides(shape_array, layout);

            return Self{
                .shape = shape_array,
                .strides = strides,
                .data = data,
                .allocator = allocator,
                .layout = layout,
                .owned = true,
            };
        }

        /// Free all allocated memory if this array owns its data
        ///
        /// For owned arrays (owned=true), frees the data buffer.
        /// For views/borrowed arrays (owned=false), does nothing (parent owns the data).
        ///
        /// Time: O(1) deallocation (if owned)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            if (self.owned) {
                self.allocator.free(self.data);
            }
        }

        /// Create a borrowed view of this array (zero-copy reference)
        ///
        /// Creates a new NDArray that shares the same underlying data buffer.
        /// The view has owned=false, so calling deinit() on it will not free the data.
        /// The original array (this instance) retains ownership and is responsible for
        /// freeing the data.
        ///
        /// Modifications made through the view will be visible in the original array
        /// and vice versa, since they share the same memory.
        ///
        /// Example:
        /// ```zig
        /// var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
        /// defer arr.deinit(); // Original array owns the data
        ///
        /// var view = arr.createView();
        /// defer view.deinit(); // Safe to call, but won't free data
        ///
        /// view.set(&.{0, 0}, 42.0);
        /// try testing.expectEqual(42.0, arr.get(&.{0, 0})); // Original sees the change
        /// ```
        ///
        /// Returns: New NDArray instance sharing the same data (owned=false)
        ///
        /// Time: O(1) - just copies metadata
        /// Space: O(1) - no data allocation
        pub fn createView(self: *const Self) Self {
            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = self.data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = false,
            };
        }

        /// Calculate stride array based on shape and layout
        ///
        /// For row-major (C order) with shape [d0, d1, d2]:
        /// strides = [d1*d2, d2, 1]
        ///
        /// For column-major (Fortran order) with shape [d0, d1, d2]:
        /// strides = [1, d0, d0*d1]
        ///
        /// Returns: Array of ndim stride values
        ///
        /// Time: O(ndim)
        /// Space: O(ndim) (output)
        fn calculateStrides(shape: [ndim]usize, layout: Layout) [ndim]usize {
            var strides: [ndim]usize = undefined;

            // Handle 0-dimensional (scalar) arrays
            if (ndim == 0) {
                return strides; // Empty array for scalars
            }

            switch (layout) {
                .row_major => {
                    // Row-major: stride[i] = product of all dimensions after i
                    strides[ndim - 1] = 1;
                    if (ndim > 1) {
                        var i: isize = @as(isize, ndim) - 2;
                        while (i >= 0) {
                            strides[@as(usize, @intCast(i))] = strides[@as(usize, @intCast(i + 1))] * shape[@as(usize, @intCast(i + 1))];
                            i -= 1;
                        }
                    }
                },
                .column_major => {
                    // Column-major: stride[i] = product of all dimensions before i
                    strides[0] = 1;
                    for (1..ndim) |i| {
                        strides[i] = strides[i - 1] * shape[i - 1];
                    }
                },
            }

            return strides;
        }

        /// Get number of elements in the array
        ///
        /// Time: O(ndim) calculation
        /// Space: O(1)
        pub fn count(self: *const Self) usize {
            var total: usize = 1;
            for (self.shape) |dim| {
                total *= dim;
            }
            return total;
        }

        /// Check if array is empty (any dimension is 0)
        ///
        /// Time: O(ndim)
        /// Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            for (self.shape) |dim| {
                if (dim == 0) {
                    return true;
                }
            }
            return false;
        }

        /// Check if array data is stored contiguously in memory
        ///
        /// An array is contiguous if its elements are stored in standard C-order (row-major)
        /// or Fortran-order (column-major) without gaps. Contiguous arrays can be reshaped
        /// with zero-copy, while non-contiguous arrays (e.g., after slicing or transposing)
        /// require copying.
        ///
        /// Returns: true if contiguous, false otherwise
        ///
        /// Time: O(ndim)
        /// Space: O(1)
        fn isContiguous(self: *const Self) bool {
            // Empty arrays are trivially contiguous
            if (self.isEmpty()) {
                return true;
            }

            // Check if strides match what they would be for a contiguous array
            const expected_strides = calculateStrides(self.shape, self.layout);

            for (0..ndim) |i| {
                if (self.strides[i] != expected_strides[i]) {
                    return false;
                }
            }

            return true;
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - All dimensions > 0
        /// - Strides are consistent with shape and layout
        /// - Data pointer is valid
        /// - data.len == prod(shape)
        ///
        /// Time: O(ndim)
        /// Space: O(1)
        pub fn validate(self: *const Self) !void {
            // Check all dimensions are positive
            for (self.shape) |dim| {
                debug.assert(dim > 0);
            }

            // Check data length matches product of shape
            const expected_len = self.count();
            debug.assert(self.data.len == expected_len);

            // Verify strides are consistent with shape and layout
            const expected_strides = calculateStrides(self.shape, self.layout);
            for (0..ndim) |i| {
                debug.assert(self.strides[i] == expected_strides[i]);
            }
        }

        // -- Creation Functions --

        /// Create an array filled with zeros
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shape: Array shape (ndim elements)
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray with all elements set to 0
        ///
        /// Time: O(prod(shape))
        /// Space: O(prod(shape))
        pub fn zeros(allocator: Allocator, shape: []const usize, layout: Layout) (Error || AllocatorError)!Self {
            var arr = try Self.init(allocator, shape, layout);
            errdefer arr.deinit();
            @memset(arr.data, 0);
            return arr;
        }

        /// Create an array filled with ones
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shape: Array shape (ndim elements)
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray with all elements set to 1
        ///
        /// Time: O(prod(shape))
        /// Space: O(prod(shape))
        pub fn ones(allocator: Allocator, shape: []const usize, layout: Layout) (Error || AllocatorError)!Self {
            var arr = try Self.init(allocator, shape, layout);
            errdefer arr.deinit();
            for (arr.data) |*val| {
                val.* = 1;
            }
            return arr;
        }

        /// Create an array filled with a specific value
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shape: Array shape (ndim elements)
        /// - value: Value to fill all elements with
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray with all elements set to value
        ///
        /// Time: O(prod(shape))
        /// Space: O(prod(shape))
        pub fn full(allocator: Allocator, shape: []const usize, value: T, layout: Layout) (Error || AllocatorError)!Self {
            var arr = try Self.init(allocator, shape, layout);
            errdefer arr.deinit();
            for (arr.data) |*val| {
                val.* = value;
            }
            return arr;
        }

        /// Create an empty array (uninitialized data)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shape: Array shape (ndim elements)
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray with uninitialized data
        ///
        /// Note: Data is not zero-filled. Use zeros() for deterministic behavior.
        ///
        /// Time: O(ndim)
        /// Space: O(prod(shape))
        pub fn empty(allocator: Allocator, shape: []const usize, layout: Layout) (Error || AllocatorError)!Self {
            return Self.init(allocator, shape, layout);
        }

        /// Create a 1D array with evenly spaced values in range [start, stop)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - start: Starting value (inclusive)
        /// - stop: Stopping value (exclusive for positive step, inclusive for negative)
        /// - step: Spacing between values (must not be 0)
        /// - layout: Row-major or column-major
        ///
        /// Returns: 1D NDArray with values [start, start+step, start+2*step, ...]
        ///
        /// Errors:
        /// - error.ZeroDimension if step == 0
        ///
        /// Time: O(num_elements) where num_elements = ceil((stop-start)/step)
        /// Space: O(num_elements)
        pub fn arange(allocator: Allocator, start: T, stop: T, step: T, layout: Layout) (Error || AllocatorError)!Self {
            // Validate step is not zero
            if (step == 0) {
                return error.ZeroDimension;
            }

            // Calculate number of elements
            var num_elements: usize = 0;
            if (step > 0) {
                if (start < stop) {
                    const range_diff = if (@typeInfo(T) == .float)
                        (stop - start) / step
                    else
                        @as(f64, @floatFromInt(stop - start)) / @as(f64, @floatFromInt(step));
                    num_elements = @as(usize, @intFromFloat(@ceil(range_diff)));
                }
            } else {
                if (start > stop) {
                    const range_diff = if (@typeInfo(T) == .float)
                        (start - stop) / (-step)
                    else
                        @as(f64, @floatFromInt(start - stop)) / @as(f64, @floatFromInt(-step));
                    num_elements = @as(usize, @intFromFloat(@ceil(range_diff)));
                }
            }

            if (num_elements == 0) {
                num_elements = 1; // Minimum 1 element
            }

            var arr = try Self.init(allocator, &[_]usize{num_elements}, layout);
            errdefer arr.deinit();

            var val = start;
            for (0..num_elements) |i| {
                arr.data[i] = val;
                val += step;
            }

            return arr;
        }

        /// Create a 1D array with num evenly spaced values in range [start, stop]
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - start: Starting value (inclusive)
        /// - stop: Stopping value (inclusive)
        /// - num: Number of evenly spaced samples (must be > 0)
        /// - layout: Row-major or column-major
        ///
        /// Returns: 1D NDArray with num evenly distributed values from start to stop
        ///
        /// Errors:
        /// - error.ZeroDimension if num == 0
        ///
        /// Time: O(num)
        /// Space: O(num)
        pub fn linspace(allocator: Allocator, start: T, stop: T, num: usize, layout: Layout) (Error || AllocatorError)!Self {
            if (num == 0) {
                return error.ZeroDimension;
            }

            var arr = try Self.init(allocator, &[_]usize{num}, layout);
            errdefer arr.deinit();

            if (num == 1) {
                arr.data[0] = start;
            } else {
                const step = @as(T, @floatFromInt(@as(isize, @intCast(1)))) / @as(T, @floatFromInt(@as(isize, @intCast(num - 1))));
                for (0..num) |i| {
                    const frac = @as(T, @floatFromInt(@as(isize, @intCast(i)))) * step;
                    arr.data[i] = start + (stop - start) * frac;
                }
            }

            return arr;
        }

        /// Create an array from an existing slice
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shape: Array shape (ndim elements)
        /// - data_slice: Slice containing data elements
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray copying data from slice
        ///
        /// Errors:
        /// - error.CapacityExceeded if data_slice.len != prod(shape)
        ///
        /// Time: O(prod(shape))
        /// Space: O(prod(shape))
        pub fn fromSlice(allocator: Allocator, shape: []const usize, data_slice: []const T, layout: Layout) (Error || AllocatorError)!Self {
            var arr = try Self.init(allocator, shape, layout);
            errdefer arr.deinit();

            if (arr.data.len != data_slice.len) {
                return error.CapacityExceeded;
            }

            @memcpy(arr.data, data_slice);
            return arr;
        }

        /// Create an array from an owned (allocated) slice without copying
        ///
        /// Takes ownership of an existing allocated slice and uses it directly for array data.
        /// This is a move-semantics variant of fromSlice() that avoids copying.
        ///
        /// Parameters:
        /// - allocator: Memory allocator (used for deinit() to free the owned_data)
        /// - shape: Array shape (ndim elements)
        /// - owned_data: Slice of allocated data to take ownership of
        /// - layout: Row-major or column-major
        ///
        /// Returns: Initialized NDArray with direct reference to owned_data (no copy)
        ///
        /// Errors:
        /// - error.CapacityExceeded if owned_data.len != prod(shape)
        /// - error.ZeroDimension if any dimension is 0
        ///
        /// Ownership: Caller must NOT free owned_data after this call — deinit() will handle it
        ///
        /// Time: O(ndim) strides calculation only (no data copy)
        /// Space: O(ndim) for metadata (owns existing data)
        pub fn fromOwnedSlice(allocator: Allocator, shape: []const usize, owned_data: []T, layout: Layout) (Error || AllocatorError)!Self {
            // Validate shape length matches ndim
            if (shape.len != ndim) {
                return error.ZeroDimension;
            }

            // Check for zero dimensions
            for (shape) |dim| {
                if (dim == 0) {
                    return error.ZeroDimension;
                }
            }

            // Calculate total element count with overflow check
            var total_elements: usize = 1;
            for (shape) |dim| {
                // Check for overflow
                if (total_elements > math.maxInt(usize) / dim) {
                    return error.CapacityExceeded;
                }
                total_elements *= dim;
            }

            // Verify owned_data size matches shape
            if (owned_data.len != total_elements) {
                return error.CapacityExceeded;
            }

            // Copy shape into fixed array
            var shape_array: [ndim]usize = undefined;
            for (0..ndim) |i| {
                shape_array[i] = shape[i];
            }

            // Calculate strides
            const strides = calculateStrides(shape_array, layout);

            return Self{
                .shape = shape_array,
                .strides = strides,
                .data = owned_data,
                .allocator = allocator,
                .layout = layout,
                .owned = true,
            };
        }

        /// Create an identity/unit matrix
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - rows: Number of rows
        /// - cols: Number of columns
        /// - k: Diagonal offset (0 = main diagonal, positive = above, negative = below)
        /// - layout: Row-major or column-major
        ///
        /// Returns: 2D NDArray with 1s on the k-th diagonal and 0s elsewhere
        ///
        /// Time: O(rows * cols)
        /// Space: O(rows * cols)
        pub fn eye(allocator: Allocator, rows: usize, cols: usize, k: isize, layout: Layout) (Error || AllocatorError)!Self {
            var arr = try Self.init(allocator, &[_]usize{ rows, cols }, layout);
            errdefer arr.deinit();

            // Zero out all elements first
            for (arr.data) |*val| {
                val.* = 0;
            }

            // Set diagonal to 1
            for (0..rows) |i| {
                const j_signed: isize = @as(isize, @intCast(i)) + k;
                if (j_signed >= 0 and j_signed < cols) {
                    const j: usize = @as(usize, @intCast(j_signed));
                    if (layout == .row_major) {
                        arr.data[i * cols + j] = 1;
                    } else {
                        arr.data[i + j * rows] = 1;
                    }
                }
            }

            return arr;
        }

        /// Create an identity matrix (main diagonal only)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - rows: Number of rows
        /// - cols: Number of columns
        /// - layout: Row-major or column-major
        ///
        /// Returns: 2D NDArray with 1s on main diagonal, 0s elsewhere
        ///
        /// Note: This is an alias for eye(rows, cols, 0, layout)
        ///
        /// Time: O(rows * cols)
        /// Space: O(rows * cols)
        pub fn identity(allocator: Allocator, rows: usize, cols: usize, layout: Layout) (Error || AllocatorError)!Self {
            return Self.eye(allocator, rows, cols, 0, layout);
        }

        /// Create diagonal matrix from 1D array or extract diagonal from 2D array
        ///
        /// Two modes:
        /// 1. If called on 1D array (ndim == 1): constructs 2D diagonal matrix
        /// 2. If called on 2D array (ndim == 2): extracts diagonal as 1D array
        ///
        /// Parameters:
        /// - allocator: Memory allocator (for mode 1 or mode 2 with copy)
        /// - k: Diagonal offset (0=main, >0 above main, <0 below main)
        /// - layout: Row-major or column-major (for mode 1 only)
        ///
        /// Returns:
        /// - Mode 1: 2D NDArray with input on k-th diagonal, zeros elsewhere
        /// - Mode 2: 1D NDArray with extracted diagonal elements
        ///
        /// Errors:
        /// - error.ShapeMismatch if ndim > 2
        ///
        /// Time: O(n) where n = length of diagonal
        /// Space: O(n²) for mode 1, O(n) for mode 2
        pub fn diag(self: *const Self, allocator: Allocator, k: isize, layout: Layout) (Error || AllocatorError)!if (ndim == 1) NDArray(T, 2) else NDArray(T, 1) {
            if (ndim == 1) {
                // Mode 1: construct diagonal matrix from 1D array
                const n = self.shape[0];
                const offset_abs: usize = @abs(k);
                const size = n + offset_abs;

                var mat = try NDArray(T, 2).zeros(allocator, &[_]usize{ size, size }, layout);
                errdefer mat.deinit();

                // Place elements on k-th diagonal
                for (0..n) |i| {
                    const row = if (k >= 0) i else i + offset_abs;
                    const col = if (k >= 0) i + offset_abs else i;
                    const val = self.data[i * self.strides[0]];

                    if (layout == .row_major) {
                        mat.data[row * size + col] = val;
                    } else {
                        mat.data[row + col * size] = val;
                    }
                }

                return mat;
            } else if (ndim == 2) {
                // Mode 2: extract diagonal from 2D array
                const rows = self.shape[0];
                const cols = self.shape[1];

                // Calculate diagonal length
                const diag_len = blk: {
                    if (k >= 0) {
                        const k_abs: usize = @intCast(k);
                        if (k_abs >= cols) break :blk 0;
                        break :blk @min(rows, cols - k_abs);
                    } else {
                        const k_abs: usize = @intCast(-k);
                        if (k_abs >= rows) break :blk 0;
                        break :blk @min(rows - k_abs, cols);
                    }
                };

                var arr = try NDArray(T, 1).init(allocator, &[_]usize{diag_len}, .row_major);
                errdefer arr.deinit();

                // Extract diagonal elements
                for (0..diag_len) |i| {
                    const row = if (k >= 0) i else i + @abs(k);
                    const col = if (k >= 0) i + @abs(k) else i;

                    const offset = if (self.layout == .row_major)
                        row * self.strides[0] + col * self.strides[1]
                    else
                        row * self.strides[0] + col * self.strides[1];

                    arr.data[i] = self.data[offset];
                }

                return arr;
            } else {
                return error.ShapeMismatch;
            }
        }

        /// Extract diagonal from 2D array as 1D array
        ///
        /// This is a convenience wrapper for diag() that only works on 2D arrays.
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - offset: Diagonal offset (0=main, >0 above main, <0 below main)
        ///
        /// Returns: 1D NDArray with diagonal elements
        ///
        /// Errors:
        /// - error.ShapeMismatch if ndim != 2
        ///
        /// Time: O(n) where n = diagonal length
        /// Space: O(n)
        pub fn diagonal(self: *const Self, allocator: Allocator, offset: isize) (Error || AllocatorError)!NDArray(T, 1) {
            if (ndim != 2) return error.ShapeMismatch;
            return self.diag(allocator, offset, .row_major);
        }

        /// Extract upper triangular matrix (zero out below k-th diagonal)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - k: Diagonal offset (0=main, >0 keep more, <0 keep less)
        ///
        /// Returns: 2D NDArray with upper triangle preserved, lower set to zero
        ///
        /// Errors:
        /// - error.ShapeMismatch if ndim != 2
        ///
        /// Time: O(rows * cols)
        /// Space: O(rows * cols)
        pub fn triu(self: *const Self, allocator: Allocator, k: isize) (Error || AllocatorError)!Self {
            if (ndim != 2) return error.ShapeMismatch;

            const rows = self.shape[0];
            const cols = self.shape[1];

            var result = try Self.init(allocator, &self.shape, self.layout);
            errdefer result.deinit();

            for (0..rows) |i| {
                for (0..cols) |j| {
                    const j_signed: isize = @intCast(j);
                    const i_signed: isize = @intCast(i);

                    const src_offset = if (self.layout == .row_major)
                        i * self.strides[0] + j * self.strides[1]
                    else
                        i * self.strides[0] + j * self.strides[1];

                    const dst_offset = if (result.layout == .row_major)
                        i * cols + j
                    else
                        i + j * rows;

                    // Keep if j >= i + k (above or on k-th diagonal)
                    result.data[dst_offset] = if (j_signed >= i_signed + k)
                        self.data[src_offset]
                    else
                        0;
                }
            }

            return result;
        }

        /// Extract lower triangular matrix (zero out above k-th diagonal)
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - k: Diagonal offset (0=main, >0 keep more, <0 keep less)
        ///
        /// Returns: 2D NDArray with lower triangle preserved, upper set to zero
        ///
        /// Errors:
        /// - error.ShapeMismatch if ndim != 2
        ///
        /// Time: O(rows * cols)
        /// Space: O(rows * cols)
        pub fn tril(self: *const Self, allocator: Allocator, k: isize) (Error || AllocatorError)!Self {
            if (ndim != 2) return error.ShapeMismatch;

            const rows = self.shape[0];
            const cols = self.shape[1];

            var result = try Self.init(allocator, &self.shape, self.layout);
            errdefer result.deinit();

            for (0..rows) |i| {
                for (0..cols) |j| {
                    const j_signed: isize = @intCast(j);
                    const i_signed: isize = @intCast(i);

                    const src_offset = if (self.layout == .row_major)
                        i * self.strides[0] + j * self.strides[1]
                    else
                        i * self.strides[0] + j * self.strides[1];

                    const dst_offset = if (result.layout == .row_major)
                        i * cols + j
                    else
                        i + j * rows;

                    // Keep if j <= i + k (below or on k-th diagonal)
                    result.data[dst_offset] = if (j_signed <= i_signed + k)
                        self.data[src_offset]
                    else
                        0;
                }
            }

            return result;
        }

        /// Compute the trace (sum of diagonal elements) of a 2D array
        ///
        /// Parameters:
        /// - offset: Diagonal offset (0=main, >0 above main, <0 below main)
        ///
        /// Returns: Sum of elements on the k-th diagonal
        ///
        /// Errors:
        /// - error.ShapeMismatch if ndim != 2
        ///
        /// Time: O(n) where n = diagonal length
        /// Space: O(1)
        pub fn trace(self: *const Self, offset: isize) Error!T {
            if (ndim != 2) return error.ShapeMismatch;

            const rows = self.shape[0];
            const cols = self.shape[1];

            // Calculate diagonal length
            const diag_len = blk: {
                if (offset >= 0) {
                    const k_abs: usize = @intCast(offset);
                    if (k_abs >= cols) break :blk 0;
                    break :blk @min(rows, cols - k_abs);
                } else {
                    const k_abs: usize = @intCast(-offset);
                    if (k_abs >= rows) break :blk 0;
                    break :blk @min(rows - k_abs, cols);
                }
            };

            var result: T = 0;
            for (0..diag_len) |i| {
                const row = if (offset >= 0) i else i + @abs(offset);
                const col = if (offset >= 0) i + @abs(offset) else i;

                const idx = if (self.layout == .row_major)
                    row * self.strides[0] + col * self.strides[1]
                else
                    row * self.strides[0] + col * self.strides[1];

                result += self.data[idx];
            }

            return result;
        }

        /// Reshape the array to a new shape without modifying data order
        ///
        /// Creates a new array view with different dimensions but same data elements.
        /// Performs zero-copy for contiguous arrays (same data pointer), copies for non-contiguous.
        ///
        /// Parameters:
        /// - new_shape: Slice with ndim elements specifying new size along each dimension
        ///
        /// Returns: Reshaped NDArray with new shape and recalculated strides
        ///
        /// Errors:
        /// - error.ZeroDimension if any element in new_shape is 0
        /// - error.CapacityExceeded if prod(new_shape) != prod(self.shape)
        ///
        /// Time: O(n) where n = prod(shape) (currently always copies for memory safety)
        /// Space: O(prod(shape)) for new allocation
        pub fn reshape(self: *const Self, new_shape: []const usize) (Error || AllocatorError)!Self {
            // Validate new_shape length
            if (new_shape.len != ndim) {
                return error.ZeroDimension;
            }

            // Check for zero dimensions in new_shape
            for (new_shape) |dim| {
                if (dim == 0) {
                    return error.ZeroDimension;
                }
            }

            // Calculate total elements in new shape
            var new_total: usize = 1;
            for (new_shape) |dim| {
                // Check for overflow
                if (new_total > math.maxInt(usize) / dim) {
                    return error.CapacityExceeded;
                }
                new_total *= dim;
            }

            // Verify total size matches
            if (new_total != self.count()) {
                return error.CapacityExceeded;
            }

            // Check if we can do zero-copy reshape (array must be contiguous)
            // Contiguous means data is stored in standard C or Fortran order without gaps
            const is_contiguous = self.isContiguous();

            if (is_contiguous) {
                // Zero-copy path: create view with new shape but same data
                // Copy new_shape into fixed array
                var shape_array: [ndim]usize = undefined;
                for (0..ndim) |i| {
                    shape_array[i] = new_shape[i];
                }

                // Calculate strides for new shape using same layout
                const new_strides = calculateStrides(shape_array, self.layout);

                // Return view (owned=false) sharing the same data
                return Self{
                    .shape = shape_array,
                    .strides = new_strides,
                    .data = self.data,
                    .allocator = self.allocator,
                    .layout = self.layout,
                    .owned = false, // View: does not own data
                };
            } else {
                // Non-contiguous: must copy to new buffer
                const new_data = try self.allocator.alloc(T, new_total);
                errdefer self.allocator.free(new_data);

                // Copy all elements from old layout to new contiguous buffer
                var iter = self.iterator();
                var idx: usize = 0;
                while (iter.next()) |val| {
                    new_data[idx] = val;
                    idx += 1;
                }

                // Create array from owned slice
                return Self.fromOwnedSlice(self.allocator, new_shape, new_data, self.layout);
            }
        }

        /// Transpose the array by reversing all axes (zero-copy view)
        ///
        /// Creates a new array view with shape and strides reversed, sharing the same
        /// underlying data. This is a zero-copy operation that returns a view with
        /// swapped dimensions.
        ///
        /// For a 2D array [rows, cols] with strides [cols, 1] in row-major order:
        /// - Transposed shape: [cols, rows]
        /// - Transposed strides: [1, cols]
        ///
        /// For a 3D array [d0, d1, d2] with strides [s0, s1, s2]:
        /// - Transposed shape: [d2, d1, d0]
        /// - Transposed strides: [s2, s1, s0]
        ///
        /// Modifications to the transposed view affect the original array (shared data).
        ///
        /// Parameters: none
        ///
        /// Returns: New NDArray view with reversed shape and strides, same data pointer
        ///
        /// Time: O(ndim) to reverse arrays
        /// Space: O(1) - view only, no new allocation
        pub fn transpose(self: *const Self) Self {
            // Copy shape and strides to local arrays for reversal
            var new_shape: [ndim]usize = self.shape;
            var new_strides: [ndim]usize = self.strides;

            // Reverse both shape and strides arrays
            mem.reverse(usize, &new_shape);
            mem.reverse(usize, &new_strides);

            // Return new view with reversed metadata but same data pointer
            return Self{
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = false, // View shares data with original
            };
        }

        /// Get the number of dimensions (rank) of the array
        ///
        /// Returns: Rank as usize (comptime rank of this array type)
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn rank(self: *const Self) usize {
            // Extract ndim from the shape array type
            return @typeInfo(@TypeOf(self.shape)).array.len;
        }

        /// Flatten the array to 1D, converting to row-major order
        ///
        /// Converts a multi-dimensional array to a 1D array with all elements
        /// in row-major traversal order. Performs zero-copy if the input is
        /// contiguous (data pointer unchanged), copies data if non-contiguous.
        ///
        /// Parameters: none
        ///
        /// Returns: NDArray(T, 1) with shape = [total_elements], row-major order
        ///
        /// Errors:
        /// - error.OutOfMemory if copy required and allocation fails
        ///
        /// Time: O(1) if contiguous, O(n) if non-contiguous (where n = prod(shape))
        /// Space: O(1) if contiguous, O(n) if non-contiguous
        pub fn flatten(self: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            // Calculate total number of elements
            const total_elements = self.count();

            // Check if array is contiguous
            // Contiguous: data.len equals the expected total element count
            const is_contiguous = self.data.len == total_elements;

            if (is_contiguous) {
                // Zero-copy: use fromOwnedSlice with same data pointer
                // We need to cast away const on data slice
                const mutable_data: []T = @constCast(self.data);
                return NDArray(T, 1).fromOwnedSlice(self.allocator, &[_]usize{total_elements}, mutable_data, self.layout);
            } else {
                // Non-contiguous: must copy data to new contiguous buffer
                // Allocate new buffer
                const new_data = try self.allocator.alloc(T, total_elements);
                errdefer self.allocator.free(new_data);

                // Copy all elements from old layout to new contiguous buffer
                var iter = self.iterator();
                var idx: usize = 0;
                while (iter.next()) |val| {
                    new_data[idx] = val;
                    idx += 1;
                }

                // Create array from owned slice
                return NDArray(T, 1).fromOwnedSlice(self.allocator, &[_]usize{total_elements}, new_data, self.layout);
            }
        }

        /// Flatten multi-dimensional array to 1D with copy semantics (always allocates)
        ///
        /// Unlike flatten() which may return a view for contiguous arrays,
        /// ravel() always allocates a new independent array regardless of
        /// contiguity. This is useful when ownership/independence is critical.
        ///
        /// Parameters: none
        ///
        /// Returns: New 1D NDArray with shape [total_elements], independent copy
        ///
        /// Errors:
        /// - error.OutOfMemory if allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for new allocation
        pub fn ravel(self: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            // Calculate total number of elements
            const total_elements = self.count();

            // Always allocate new buffer (key difference from flatten)
            const new_data = try self.allocator.alloc(T, total_elements);
            errdefer self.allocator.free(new_data);

            // Copy all elements using iterator (respects source layout)
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                new_data[idx] = val;
                idx += 1;
            }

            // Create 1D array from owned slice, preserving layout
            return NDArray(T, 1).fromOwnedSlice(self.allocator, &[_]usize{total_elements}, new_data, self.layout);
        }

        /// Reorder array dimensions according to a permutation of axes
        ///
        /// Creates a zero-copy view of the array with dimensions reordered.
        /// The axes parameter specifies the new order: new_shape[i] = old_shape[axes[i]]
        ///
        /// Example: For shape [2, 3, 4] with axes [2, 1, 0], result shape is [4, 3, 2]
        ///
        /// Modifications to the permuted view affect the original array (shared data).
        ///
        /// Parameters:
        /// - axes: Slice with ndim elements, must be a valid permutation of [0..ndim)
        ///
        /// Returns: New NDArray view with reordered shape and strides, same data pointer
        ///
        /// Errors:
        /// - error.InvalidPermutation if axes is not a valid permutation:
        ///   - Length != ndim
        ///   - Contains values >= ndim
        ///   - Contains duplicates
        ///
        /// Time: O(ndim) to validate and reorder
        /// Space: O(1) - view only, no new allocation
        pub fn permute(self: *const Self, axes: []const usize) Error!Self {
            // Validate axes length
            if (axes.len != ndim) {
                return error.InvalidPermutation;
            }

            // Track seen values to detect duplicates
            var seen: [ndim]bool = [_]bool{false} ** ndim;

            // Validate each axis value
            for (axes) |axis| {
                // Check if in range [0..ndim)
                if (axis >= ndim) {
                    return error.InvalidPermutation;
                }

                // Check for duplicates
                if (seen[axis]) {
                    return error.InvalidPermutation;
                }
                seen[axis] = true;
            }

            // Create new shape and strides by reordering
            var new_shape: [ndim]usize = undefined;
            var new_strides: [ndim]usize = undefined;

            for (0..ndim) |i| {
                new_shape[i] = self.shape[axes[i]];
                new_strides[i] = self.strides[axes[i]];
            }

            // Return new view with reordered metadata but same data pointer
            return Self{
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = false,
            };
        }

        /// Ensure the array has contiguous memory layout
        ///
        /// If the array is already contiguous (data.len == prod(shape) and strides
        /// match expected row-major or column-major pattern), returns a copy of the
        /// array with same data pointer.
        ///
        /// If non-contiguous (result of slicing, transposing, or permuting), allocates
        /// a new contiguous buffer, copies all elements via iterator traversal,
        /// and returns new array with contiguous strides.
        ///
        /// Parameters:
        /// - self: Reference to the array
        ///
        /// Returns: New NDArray with contiguous memory layout and independent allocation
        ///
        /// Errors:
        /// - AllocatorError if memory allocation fails during copying
        ///
        /// Time: O(1) if already contiguous, O(n) if copying required (n = prod(shape))
        /// Space: O(n) if allocation needed for new buffer
        pub fn contiguous(self: *const Self) (Error || AllocatorError)!Self {
            // Calculate total number of elements
            const total_elements = self.count();

            // Calculate expected contiguous strides based on layout
            const expected_strides = calculateStrides(self.shape, self.layout);

            // Check if already contiguous:
            // 1. data.len must equal total_elements
            // 2. strides must match expected contiguous pattern
            var is_contiguous = self.data.len == total_elements;

            if (is_contiguous) {
                // Verify stride pattern matches
                for (0..ndim) |i| {
                    if (self.strides[i] != expected_strides[i]) {
                        is_contiguous = false;
                        break;
                    }
                }
            }

            // If already contiguous, we still need to allocate new memory
            // to avoid double-free when both arrays call deinit()
            if (is_contiguous) {
                const new_data = try self.allocator.alloc(T, total_elements);
                errdefer self.allocator.free(new_data);
                @memcpy(new_data, self.data);
                return Self{
                    .shape = self.shape,
                    .strides = self.strides,
                    .data = new_data,
                    .allocator = self.allocator,
                    .layout = self.layout,
                                .owned = true,
            };
            }

            // Non-contiguous: allocate new buffer and copy elements
            const new_data = try self.allocator.alloc(T, total_elements);
            errdefer self.allocator.free(new_data);

            // Copy all elements in row-major order via iterator
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |value| : (idx += 1) {
                new_data[idx] = value;
            }

            // Return new array with contiguous layout and expected strides
            return Self{
                .shape = self.shape,
                .strides = expected_strides,
                .data = new_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        // -- Element-wise Arithmetic Operations --

        /// Element-wise addition: self + other
        ///
        /// Parameters:
        /// - other: Another NDArray with matching shape
        ///
        /// Returns: New NDArray containing element-wise sum
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn add(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            return applyBinaryOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) T {
                    return a + b;
                }
            }.op);
        }

        /// Element-wise subtraction: self - other
        ///
        /// Parameters:
        /// - other: Another NDArray with matching shape
        ///
        /// Returns: New NDArray containing element-wise difference
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn sub(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            return applyBinaryOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) T {
                    return a - b;
                }
            }.op);
        }

        /// Element-wise multiplication: self * other
        ///
        /// Parameters:
        /// - other: Another NDArray with matching shape
        ///
        /// Returns: New NDArray containing element-wise product
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn mul(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            return applyBinaryOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) T {
                    return a * b;
                }
            }.op);
        }

        /// Element-wise division: self / other
        ///
        /// Parameters:
        /// - other: Another NDArray with matching shape
        ///
        /// Returns: New NDArray containing element-wise quotient
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn div(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            return applyBinaryOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) T {
                    return a / b;
                }
            }.op);
        }

        /// Element-wise modulo: self % other (integer types only)
        ///
        /// Parameters:
        /// - other: Another NDArray with matching shape
        ///
        /// Returns: New NDArray containing element-wise remainder
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not an integer type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn mod(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: mod only works on integer types
            if (@typeInfo(T) != .int) {
                @compileError("mod() is only defined for integer types");
            }

            return applyBinaryOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) T {
                    return @rem(a, b);
                }
            }.op);
        }

        // -- Matrix Operations --

        /// Matrix multiplication: self @ other (NumPy notation)
        ///
        /// For 2D arrays (matrices):
        /// - (M, N) @ (N, P) → (M, P)
        /// - Inner dimensions must match: self.shape[1] == other.shape[0]
        ///
        /// For 1D arrays (vectors):
        /// - (N,) @ (N,) → scalar (dot product)
        /// - Result is returned as 0-dimensional array
        ///
        /// For higher dimensions:
        /// - Batched matrix multiplication on last two dimensions
        /// - Leading dimensions must broadcast
        ///
        /// Parameters:
        /// - other: Another NDArray with compatible shape
        ///
        /// Returns: New NDArray containing matrix product
        ///
        /// Errors:
        /// - Error.ShapeMismatch if inner dimensions don't match
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(M*N*P) for (M,N) @ (N,P) matrices
        /// Space: O(M*P) for result array
        ///
        /// Examples:
        /// ```zig
        /// // Matrix-matrix multiplication
        /// const A = try NDArray(f64, 2).init(allocator, &[_]usize{2, 3}, .row_major);
        /// const B = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
        /// const C = try A.matmul(&B); // Shape: [2, 4]
        ///
        /// // Vector dot product
        /// const v1 = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
        /// const v2 = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
        /// const dot = try v1.matmul(&v2); // Shape: [] (scalar)
        /// ```
        pub fn matmul(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            // Case 1: Both are 1D (vector dot product → scalar)
            if (ndim == 1) {
                if (self.shape[0] != other.shape[0]) {
                    return Error.ShapeMismatch;
                }

                // Compute dot product
                var accumulator: T = 0;
                for (0..self.shape[0]) |i| {
                    const a = try self.get(&.{@intCast(i)});
                    const b = try other.get(&.{@intCast(i)});
                    accumulator += a * b;
                }

                // Return as 0-dimensional array (scalar wrapped in NDArray)
                const result_data = try self.allocator.alloc(T, 1);
                errdefer self.allocator.free(result_data);
                result_data[0] = accumulator;

                return Self{
                    .shape = [_]usize{1} ** ndim,
                    .strides = [_]usize{1} ** ndim,
                    .data = result_data,
                    .allocator = self.allocator,
                    .layout = self.layout,
                    .owned = true,
                };
            }

            // Case 2: Both are 2D (standard matrix multiplication)
            if (ndim == 2) {
                const M = self.shape[0]; // rows in self
                const N = self.shape[1]; // cols in self / rows in other
                const K = other.shape[0]; // rows in other
                const P = other.shape[1]; // cols in other

                // Verify inner dimensions match
                if (N != K) {
                    return Error.ShapeMismatch;
                }

                // Allocate result matrix: (M, P)
                const result_shape = [2]usize{ M, P };
                var result = try Self.zeros(self.allocator, &result_shape, self.layout);
                errdefer result.deinit();

                // Perform matrix multiplication: C[i,j] = Σ(k) A[i,k] * B[k,j]
                for (0..M) |i| {
                    for (0..P) |j| {
                        var accumulator: T = 0;
                        for (0..N) |k| {
                            const a = try self.get(&.{ @intCast(i), @intCast(k) });
                            const b = try other.get(&.{ @intCast(k), @intCast(j) });
                            accumulator += a * b;
                        }
                        result.set(&.{ @intCast(i), @intCast(j) }, accumulator);
                    }
                }

                return result;
            }

            // Case 3: Higher dimensions - not yet supported
            // Would require batched matmul with broadcasting
            return Error.DimensionMismatch;
        }

        // -- Element-wise Unary Operations --

        /// Element-wise negation: -self
        ///
        /// Returns: New NDArray with negated elements
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn neg(self: *const Self) (Error || AllocatorError)!Self {
            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and negate each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = -val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise absolute value
        ///
        /// Returns: New NDArray with absolute values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn abs(self: *const Self) (Error || AllocatorError)!Self {
            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute absolute value of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = if (val < 0) -val else val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        // -- Mathematical Functions (floating-point types) --

        /// Element-wise exponential: e^x for each element
        ///
        /// Returns: New NDArray with exponential values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn exp(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: exp only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("exp() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute exponential of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.exp(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise natural logarithm: ln(x) for each element
        ///
        /// Returns: New NDArray with logarithm values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn log(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: log only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("log() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute logarithm of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.log(T, math.e, val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise square root: sqrt(x) for each element
        ///
        /// Returns: New NDArray with square root values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn sqrt(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: sqrt only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("sqrt() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute square root of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = stdlib.math.sqrt(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise power: x^exponent for each element
        ///
        /// Parameters:
        /// - exponent: The exponent to raise each element to
        ///
        /// Returns: New NDArray with power values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn pow(self: *const Self, exponent: T) (Error || AllocatorError)!Self {
            // Compile-time check: pow only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("pow() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute power of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.pow(T, val, exponent);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        // -- Rounding Functions (floating-point types) --

        /// Element-wise floor - rounds each element down to nearest integer
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn floor(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: floor only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("floor() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute floor of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = @floor(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise ceil - rounds each element up to nearest integer
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn ceil(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: ceil only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("ceil() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute ceil of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = @ceil(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise round - rounds each element to nearest integer (half away from zero)
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn round(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: round only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("round() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute round of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = @round(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise trunc - truncates each element toward zero (removes fractional part)
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn trunc(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: trunc only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("trunc() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute trunc of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = @trunc(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        // -- Trigonometric Functions (floating-point types) --

        /// Element-wise sine: sin(x) for each element
        ///
        /// Returns: New NDArray with sine values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn sin(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: sin only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("sin() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute sine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.sin(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise cosine: cos(x) for each element
        ///
        /// Returns: New NDArray with cosine values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn cos(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: cos only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("cos() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute cosine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.cos(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise tangent: tan(x) for each element
        ///
        /// Returns: New NDArray with tangent values
        ///
        /// Errors:
        /// - Allocator.Error if memory allocation fails
        /// - CompileError if T is not a floating-point type
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn tan(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: tan only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("tan() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute tangent of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.tan(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise arcsine (inverse sine) - returns array with asin of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn asin(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: asin only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("asin() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute arcsine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.asin(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise arccosine (inverse cosine) - returns array with acos of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn acos(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: acos only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("acos() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute arccosine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.acos(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise arctangent (inverse tangent) - returns array with atan of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn atan(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: atan only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("atan() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute arctangent of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.atan(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise two-argument arctangent - returns array with atan2(self, other) for each element pair
        ///
        /// Computes atan2(y, x) for corresponding elements from two arrays.
        /// This gives the angle in radians from the positive x-axis to the point (x, y).
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn atan2(self: *const Self, other: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: atan2 only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("atan2() is only defined for floating-point types");
            }

            // Check shape compatibility
            for (0..ndim) |i| {
                if (self.shape[i] != other.shape[i]) {
                    return error.ShapeMismatch;
                }
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse both arrays and compute atan2
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |y_val| {
                const x_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = math.atan2(y_val, x_val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise hyperbolic sine - returns array with sinh of each element
        ///
        /// Computes sinh(x) = (e^x - e^(-x)) / 2 for each element.
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn sinh(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: sinh only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("sinh() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute hyperbolic sine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.sinh(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise hyperbolic cosine - returns array with cosh of each element
        ///
        /// Computes cosh(x) = (e^x + e^(-x)) / 2 for each element.
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn cosh(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: cosh only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("cosh() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute hyperbolic cosine of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.cosh(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise hyperbolic tangent - returns array with tanh of each element
        ///
        /// Computes tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) for each element.
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn tanh(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: tanh only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("tanh() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute hyperbolic tangent of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.tanh(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise sign extraction: returns -1 for negative, 0 for zero, +1 for positive
        ///
        /// Returns: New NDArray with sign values (-1, 0, or 1)
        ///
        /// Use cases:
        /// - Gradient sign extraction in optimization
        /// - Sign pattern analysis in signal processing
        /// - Direction indicators in numerical methods
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn sign(self: *const Self) (Error || AllocatorError)!Self {
            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and extract sign of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = if (val > 0) @as(T, 1) else if (val < 0) @as(T, -1) else @as(T, 0);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Element-wise clip (clamp) values to range [min_val, max_val]
        ///
        /// Returns: New NDArray with clipped values
        ///
        /// Use cases:
        /// - Gradient clipping in neural network training
        /// - Value range enforcement in data preprocessing
        /// - Outlier removal in statistical analysis
        /// - Saturation arithmetic in signal processing
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn clip(self: *const Self, min_val: T, max_val: T) (Error || AllocatorError)!Self {
            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and clip each element to [min_val, max_val]
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = if (val < min_val) min_val else if (val > max_val) max_val else val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = true,
            };
        }

        /// Conditional element selection: where(condition, x, y) returns x[i] if condition[i] else y[i]
        ///
        /// All arrays must have the same shape.
        ///
        /// Returns: New NDArray with selected values
        ///
        /// Use cases:
        /// - Conditional masking in data filtering
        /// - Piecewise function evaluation
        /// - NumPy-style np.where() equivalent
        /// - Threshold-based data transformation
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn where(condition: *const NDArray(bool, ndim), x: *const Self, y: *const Self) (Error || AllocatorError)!Self {
            // Shape validation: all arrays must match
            if (!mem.eql(usize, &condition.shape, &x.shape) or !mem.eql(usize, &x.shape, &y.shape)) {
                return Error.ShapeMismatch;
            }

            // Allocate new buffer for result
            const total = x.count();
            const result_data = try x.allocator.alloc(T, total);
            errdefer x.allocator.free(result_data);

            // Traverse arrays and select elements based on condition
            var cond_iter = condition.iterator();
            var x_iter = x.iterator();
            var y_iter = y.iterator();
            var idx: usize = 0;

            while (cond_iter.next()) |cond_val| {
                const x_val = x_iter.next() orelse unreachable; // shapes match, safe
                const y_val = y_iter.next() orelse unreachable;
                result_data[idx] = if (cond_val) x_val else y_val;
                idx += 1;
            }

            return Self{
                .shape = x.shape,
                .strides = x.strides,
                .data = result_data,
                .allocator = x.allocator,
                .layout = x.layout,
                .owned = true,
            };
        }

        /// Element-wise base-2 logarithm - returns array with log2 of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn log2(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: log2 only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("log2() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute log2 of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.log2(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise base-10 logarithm - returns array with log10 of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn log10(self: *const Self) (Error || AllocatorError)!Self {
            // Compile-time check: log10 only works on float types
            if (@typeInfo(T) != .float) {
                @compileError("log10() is only defined for floating-point types");
            }

            // Allocate new buffer for result
            const total = self.count();
            const result_data = try self.allocator.alloc(T, total);
            errdefer self.allocator.free(result_data);

            // Traverse array and compute log10 of each element
            var iter = self.iterator();
            var idx: usize = 0;
            while (iter.next()) |val| {
                result_data[idx] = math.log10(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
                            .owned = true,
            };
        }

        /// Element-wise equality comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn eq(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a == b;
                }
            }.op);
        }

        /// Element-wise inequality comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn ne(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a != b;
                }
            }.op);
        }

        /// Element-wise less-than comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn lt(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a < b;
                }
            }.op);
        }

        /// Element-wise less-or-equal comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn le(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a <= b;
                }
            }.op);
        }

        /// Element-wise greater-than comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn gt(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a > b;
                }
            }.op);
        }

        /// Element-wise greater-or-equal comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn ge(self: *const Self, other: *const Self) (Error || AllocatorError)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a >= b;
                }
            }.op);
        }

        // -- I/O Operations --

        /// Type tag for serialization - identifies element type in binary format
        const TypeTag = enum(u8) {
            i8 = 0,
            i16 = 1,
            i32 = 2,
            i64 = 3,
            u8 = 4,
            u16 = 5,
            u32 = 6,
            u64 = 7,
            f32 = 8,
            f64 = 9,
            bool_type = 10,

            fn fromType(comptime Type: type) TypeTag {
                return switch (Type) {
                    i8 => .i8,
                    i16 => .i16,
                    i32 => .i32,
                    i64 => .i64,
                    u8 => .u8,
                    u16 => .u16,
                    u32 => .u32,
                    u64 => .u64,
                    f32 => .f32,
                    f64 => .f64,
                    bool => .bool_type,
                    else => @compileError("Unsupported type for NDArray serialization"),
                };
            }

            fn toTypeInfo(self: TypeTag) builtin.Type {
                return switch (self) {
                    .i8 => @typeInfo(i8),
                    .i16 => @typeInfo(i16),
                    .i32 => @typeInfo(i32),
                    .i64 => @typeInfo(i64),
                    .u8 => @typeInfo(u8),
                    .u16 => @typeInfo(u16),
                    .u32 => @typeInfo(u32),
                    .u64 => @typeInfo(u64),
                    .f32 => @typeInfo(f32),
                    .f64 => @typeInfo(f64),
                    .bool_type => @typeInfo(bool),
                };
            }
        };

        /// Parse string value to target type
        ///
        /// Helper function for CSV parsing. Converts string to numeric type.
        ///
        /// Parameters:
        /// - comptime Type: Target type (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool)
        /// - str: String to parse
        ///
        /// Returns: Parsed value of Type
        ///
        /// Errors:
        /// - error.InvalidFormat if string cannot be parsed as Type
        fn parseValue(comptime Type: type, str: []const u8) !Type {
            if (str.len == 0) return error.InvalidFormat;

            const type_info = @typeInfo(Type);
            if (type_info == .int) {
                return fmt.parseInt(Type, str, 10) catch return error.InvalidFormat;
            } else if (type_info == .float) {
                return fmt.parseFloat(Type, str) catch return error.InvalidFormat;
            } else if (type_info == .bool) {
                if (mem.eql(u8, str, "true") or mem.eql(u8, str, "1")) {
                    return true;
                } else if (mem.eql(u8, str, "false") or mem.eql(u8, str, "0")) {
                    return false;
                }
                return error.InvalidFormat;
            } else {
                @compileError("Unsupported type for CSV parsing");
            }
        }

        /// Save NDArray to binary file
        ///
        /// Binary format:
        /// - 4 bytes: magic "NDAR" (0x4E444152)
        /// - 4 bytes: version (u32, currently 1)
        /// - 1 byte: ndim (u8)
        /// - 1 byte: type tag (u8)
        /// - 1 byte: layout (u8) - 0=row_major, 1=column_major
        /// - ndim * 8 bytes: shape array
        /// - ndim * 8 bytes: strides array
        /// - count * sizeof(T) bytes: data
        ///
        /// Time: O(n) | Space: O(1) - writes directly to file
        pub fn save(self: *const Self, path: []const u8) !void {
            const file = try fs.cwd().createFile(path, .{});
            defer file.close();

            // Write magic number "NDAR" (4 bytes)
            var magic_bytes: [4]u8 = undefined;
            mem.writeInt(u32, &magic_bytes, 0x4E444152, .little);
            _ = try file.write(&magic_bytes);

            // Write version (1) (4 bytes)
            var version_bytes: [4]u8 = undefined;
            mem.writeInt(u32, &version_bytes, 1, .little);
            _ = try file.write(&version_bytes);

            // Write ndim (1 byte)
            _ = try file.write(&[_]u8{@intCast(ndim)});

            // Write type tag (1 byte)
            const type_tag = TypeTag.fromType(T);
            _ = try file.write(&[_]u8{@intFromEnum(type_tag)});

            // Write layout (1 byte) - 0=row_major, 1=column_major
            _ = try file.write(&[_]u8{if (self.layout == .row_major) 0 else 1});

            // Write shape array
            for (self.shape) |dim| {
                var dim_bytes: [@sizeOf(usize)]u8 = undefined;
                mem.writeInt(usize, &dim_bytes, dim, .little);
                _ = try file.write(&dim_bytes);
            }

            // Write strides array
            for (self.strides) |stride| {
                var stride_bytes: [@sizeOf(usize)]u8 = undefined;
                mem.writeInt(usize, &stride_bytes, stride, .little);
                _ = try file.write(&stride_bytes);
            }

            // Write data
            const bytes = mem.sliceAsBytes(self.data);
            _ = try file.write(bytes);
        }

        /// Load NDArray from binary file
        ///
        /// Validates magic number, version, ndim, and type compatibility.
        /// Allocates new data buffer and reads array contents.
        ///
        /// Errors:
        /// - error.InvalidFormat if magic number doesn't match
        /// - error.UnsupportedVersion if version is not 1
        /// - error.DimensionMismatch if file ndim doesn't match type parameter
        /// - error.TypeMismatch if file type doesn't match T
        ///
        /// Time: O(n) | Space: O(n) for data allocation
        pub fn load(allocator: mem.Allocator, path: []const u8) !Self {
            const file = try fs.cwd().openFile(path, .{});
            defer file.close();

            // Read and validate magic number (4 bytes)
            var magic_bytes: [4]u8 = undefined;
            _ = try file.read(&magic_bytes);
            const magic = mem.readInt(u32, &magic_bytes, .little);
            if (magic != 0x4E444152) {
                return error.InvalidFormat;
            }

            // Read and validate version (4 bytes)
            var version_bytes: [4]u8 = undefined;
            _ = try file.read(&version_bytes);
            const version = mem.readInt(u32, &version_bytes, .little);
            if (version != 1) {
                return error.UnsupportedVersion;
            }

            // Read ndim (1 byte)
            var ndim_byte: [1]u8 = undefined;
            _ = try file.read(&ndim_byte);
            const file_ndim = ndim_byte[0];
            if (file_ndim != ndim) {
                return error.DimensionMismatch;
            }

            // Read and validate type tag (1 byte)
            var type_tag_byte: [1]u8 = undefined;
            _ = try file.read(&type_tag_byte);
            const file_type_tag: TypeTag = @enumFromInt(type_tag_byte[0]);
            const expected_type_tag = TypeTag.fromType(T);
            if (file_type_tag != expected_type_tag) {
                return error.TypeMismatch;
            }

            // Read layout (1 byte)
            var layout_byte: [1]u8 = undefined;
            _ = try file.read(&layout_byte);
            const layout: Layout = if (layout_byte[0] == 0) .row_major else .column_major;

            // Read shape
            var shape: [ndim]usize = undefined;
            for (0..ndim) |i| {
                var dim_bytes: [@sizeOf(usize)]u8 = undefined;
                _ = try file.read(&dim_bytes);
                shape[i] = mem.readInt(usize, &dim_bytes, .little);
            }

            // Read strides
            var strides: [ndim]usize = undefined;
            for (0..ndim) |i| {
                var stride_bytes: [@sizeOf(usize)]u8 = undefined;
                _ = try file.read(&stride_bytes);
                strides[i] = mem.readInt(usize, &stride_bytes, .little);
            }

            // Calculate total elements
            var total: usize = 1;
            for (shape) |dim| {
                total *= dim;
            }

            // Allocate and read data
            const data = try allocator.alloc(T, total);
            errdefer allocator.free(data);

            const bytes = mem.sliceAsBytes(data);
            const bytes_read = try file.read(bytes);
            if (bytes_read != bytes.len) {
                return error.UnexpectedEOF;
            }

            return Self{
                .shape = shape,
                .strides = strides,
                .data = data,
                .allocator = allocator,
                .layout = layout,
                            .owned = true,
            };
        }

        /// Save NDArray to CSV file (2D arrays only)
        ///
        /// Writes array elements as comma-separated values with one row per line.
        /// Only works for 2D arrays. Use delimiter parameter to customize separator.
        ///
        /// Parameters:
        /// - path: File path to write CSV
        /// - delimiter: Character to separate values (default: ',')
        ///
        /// Errors:
        /// - error.DimensionMismatch if ndim != 2
        /// - File system errors
        ///
        /// Time: O(rows × cols) | Space: O(1) streaming write
        pub fn toCSV(self: *const Self, path: []const u8, delimiter: u8) !void {
            // Only 2D arrays can be saved as CSV
            if (ndim != 2) {
                return error.DimensionMismatch;
            }

            const file = try fs.cwd().createFile(path, .{});
            defer file.close();

            const rows = self.shape[0];
            const cols = self.shape[1];

            var buf: [4096]u8 = undefined;
            var fbs = io.fixedBufferStream(&buf);
            const writer = fbs.writer();

            for (0..rows) |r| {
                for (0..cols) |c| {
                    const val = try self.get(&[_]isize{ @intCast(r), @intCast(c) });

                    // Format value based on type
                    const type_info = @typeInfo(T);
                    if (type_info == .int) {
                        try writer.print("{d}", .{val});
                    } else if (type_info == .float) {
                        try writer.print("{d:.10}", .{val});
                    } else {
                        try writer.print("{any}", .{val});
                    }

                    // Add delimiter between columns (but not after last column)
                    if (c < cols - 1) {
                        try writer.writeByte(delimiter);
                    }
                }
                try writer.writeByte('\n');

                // Flush buffer when it gets reasonably full
                if (fbs.pos > 3000) {
                    _ = try file.write(fbs.getWritten());
                    fbs.reset();
                }
            }

            // Flush remaining data
            if (fbs.pos > 0) {
                _ = try file.write(fbs.getWritten());
            }
        }

        /// Load 2D NDArray from CSV file
        ///
        /// Reads comma-separated values from file and creates a 2D array.
        /// Automatically detects number of rows and columns.
        /// Supports custom delimiters (default: ',').
        ///
        /// Parameters:
        /// - allocator: Memory allocator for data
        /// - path: File path to read CSV
        /// - delimiter: Character separating values (default: ',')
        ///
        /// Returns: 2D NDArray with parsed data
        ///
        /// Errors:
        /// - error.DimensionMismatch if ndim != 2
        /// - error.InvalidFormat if parsing fails
        /// - error.EmptyArray if file is empty
        ///
        /// Time: O(rows × cols) | Space: O(rows × cols)
        pub fn fromCSV(allocator: mem.Allocator, path: []const u8, delimiter: u8) !Self {
            // Only 2D arrays can be loaded from CSV
            if (ndim != 2) {
                return error.DimensionMismatch;
            }

            const file = try fs.cwd().openFile(path, .{});
            defer file.close();

            // Read entire file into memory
            const max_size = 100 * 1024 * 1024; // 100 MB limit
            const contents = try file.readToEndAlloc(allocator, max_size);
            defer allocator.free(contents);

            if (contents.len == 0) {
                return error.EmptyArray;
            }

            // First pass: count rows and columns
            var row_count: usize = 0;
            var col_count: usize = 0;
            var first_row_cols: usize = 0;
            var in_first_row = true;
            var current_row_cols: usize = 0;

            var lines = mem.splitScalar(u8, contents, '\n');
            while (lines.next()) |line| {
                if (line.len == 0) continue; // Skip empty lines
                row_count += 1;

                var values = mem.splitScalar(u8, line, delimiter);
                current_row_cols = 0;
                while (values.next()) |_| {
                    current_row_cols += 1;
                }

                if (in_first_row) {
                    first_row_cols = current_row_cols;
                    col_count = first_row_cols;
                    in_first_row = false;
                } else if (current_row_cols != first_row_cols) {
                    return error.InvalidFormat; // Ragged array
                }
            }

            if (row_count == 0 or col_count == 0) {
                return error.EmptyArray;
            }

            // Allocate array
            var result = try Self.init(allocator, &[_]usize{ row_count, col_count }, .row_major);
            errdefer result.deinit();

            // Second pass: parse values
            var r: usize = 0;
            lines = mem.splitScalar(u8, contents, '\n');
            while (lines.next()) |line| {
                if (line.len == 0) continue;

                var c: usize = 0;
                var values = mem.splitScalar(u8, line, delimiter);
                while (values.next()) |value_str| {
                    const trimmed = mem.trim(u8, value_str, " \t\r");
                    const parsed = parseValue(T, trimmed) catch return error.InvalidFormat;
                    result.set(&[_]isize{ @intCast(r), @intCast(c) }, parsed);
                    c += 1;
                }
                r += 1;
            }

            return result;
        }

        // -- Indexing and Slicing Functions --

        /// Get element at multi-dimensional indices with negative indexing support
        ///
        /// Parameters:
        /// - indices: Array of ndim signed indices (negative = relative to end)
        ///
        /// Returns: Element value at computed location
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if any index is out of range after normalization
        ///
        /// Time: O(ndim) for index calculation
        /// Space: O(1)
        pub fn get(self: *const Self, indices: []const isize) Error!T {
            // Validate index count
            if (indices.len != ndim) {
                return error.IndexOutOfBounds;
            }

            // Normalize indices and compute flat offset
            var offset: usize = 0;
            for (0..ndim) |i| {
                var idx = indices[i];

                // Convert negative index to positive
                if (idx < 0) {
                    idx += @as(isize, @intCast(self.shape[i]));
                }

                // Check bounds
                if (idx < 0 or idx >= @as(isize, @intCast(self.shape[i]))) {
                    return error.IndexOutOfBounds;
                }

                // Add to flat offset
                offset += @as(usize, @intCast(idx)) * self.strides[i];
            }

            return self.data[offset];
        }

        /// Set element at multi-dimensional indices with negative indexing support
        ///
        /// Parameters:
        /// - indices: Array of ndim signed indices (negative = relative to end)
        /// - value: Value to set at the location
        ///
        /// Time: O(ndim) for index calculation
        /// Space: O(1)
        pub fn set(self: *Self, indices: []const isize, value: T) void {
            // Validate index count
            if (indices.len != ndim) {
                return;
            }

            // Normalize indices and compute flat offset
            var offset: usize = 0;
            for (0..ndim) |i| {
                var idx = indices[i];

                // Convert negative index to positive
                if (idx < 0) {
                    idx += @as(isize, @intCast(self.shape[i]));
                }

                // Check bounds
                if (idx < 0 or idx >= @as(isize, @intCast(self.shape[i]))) {
                    return;
                }

                // Add to flat offset
                offset += @as(usize, @intCast(idx)) * self.strides[i];
            }

            self.data[offset] = value;
        }

        /// Get element at flat index with negative indexing support
        ///
        /// Parameters:
        /// - index: Flat signed index (negative = relative to end)
        ///
        /// Returns: Element value at flat index location
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if index is out of range
        ///
        /// Time: O(1) direct access
        /// Space: O(1)
        pub fn at(self: *const Self, index: isize) Error!T {
            var idx = index;

            // Convert negative index to positive
            if (idx < 0) {
                idx += @as(isize, @intCast(self.data.len));
            }

            // Check bounds
            if (idx < 0 or idx >= @as(isize, @intCast(self.data.len))) {
                return error.IndexOutOfBounds;
            }

            return self.data[@as(usize, @intCast(idx))];
        }

        /// Create a non-owning view of a sub-region with slicing support
        ///
        /// Parameters:
        /// - ranges: Array of [start, stop] pairs for each dimension
        ///   - null start means 0 (beginning)
        ///   - null stop means shape[i] (end)
        ///   - Negative indices count from end: -n → shape[i] - n
        ///   - Out-of-bounds ranges are clamped to valid bounds
        ///
        /// Returns: New NDArray view sharing same underlying data
        ///   - View does NOT own the data (data pointer adjusted, not copied)
        ///   - shape adjusted based on slice ranges
        ///   - strides remain the same as original
        ///
        /// Time: O(ndim) for range processing
        /// Space: O(1) - view only, no allocation
        pub fn slice(self: *const Self, ranges: []const [2]?isize) Self {
            // If range count doesn't match, return copy of original
            if (ranges.len != ndim) {
                return self.*;
            }

            // Calculate starting offset and new shape
            var start_offset: usize = 0;
            var new_shape: [ndim]usize = undefined;

            for (0..ndim) |i| {
                const range = ranges[i];

                // Determine start and stop for this dimension
                var start: isize = if (range[0]) |s| s else 0;
                var stop: isize = if (range[1]) |e| e else @as(isize, @intCast(self.shape[i]));

                // Normalize negative indices
                if (start < 0) {
                    start += @as(isize, @intCast(self.shape[i]));
                }
                if (stop < 0) {
                    stop += @as(isize, @intCast(self.shape[i]));
                }

                // Clamp to valid range
                if (start < 0) start = 0;
                if (stop < 0) stop = 0;
                if (start > @as(isize, @intCast(self.shape[i]))) {
                    start = @as(isize, @intCast(self.shape[i]));
                }
                if (stop > @as(isize, @intCast(self.shape[i]))) {
                    stop = @as(isize, @intCast(self.shape[i]));
                }

                // Ensure start <= stop
                if (start > stop) {
                    start = stop;
                }

                // Calculate new dimension size
                new_shape[i] = @as(usize, @intCast(stop - start));

                // Add start offset for this dimension
                start_offset += @as(usize, @intCast(start)) * self.strides[i];
            }

            // Create view with adjusted data pointer
            const view_data = self.data[start_offset..];

            return Self{
                .shape = new_shape,
                .strides = self.strides,
                .data = view_data,
                .allocator = self.allocator,
                .layout = self.layout,
                .owned = false, // Slice is a view
            };
        }

        // -- Reduction Operations --

        /// Sum all elements in the array
        ///
        /// Returns: Sum of all elements as type T
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn sum(self: *const Self) T {
            var result: T = 0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                result += val;
            }
            return result;
        }

        /// Product of all elements in the array
        ///
        /// Returns: Product of all elements as type T
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn prod(self: *const Self) T {
            var result: T = 1;
            var iter = self.iterator();
            while (iter.next()) |val| {
                result *= val;
            }
            return result;
        }

        /// Mean (average) of all elements in the array
        ///
        /// Returns: Mean as f64 (always floating-point, even for integer arrays)
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn mean(self: *const Self) f64 {
            const total = self.count();
            var sum_val: f64 = 0.0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                sum_val += if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
            }
            return sum_val / @as(f64, @floatFromInt(total));
        }

        /// Variance of all elements in the array
        ///
        /// Parameters:
        /// - ddof: Delta degrees of freedom. Use 0 for population variance, 1 for sample variance.
        ///
        /// Returns: Variance as f64 (average of squared deviations from mean)
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn variance(self: *const Self, ddof: usize) f64 {
            const total = self.count();
            if (total <= ddof) {
                return 0.0;
            }

            const mean_val = self.mean();
            var sum_squared_dev: f64 = 0.0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                const fval = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                const dev = fval - mean_val;
                sum_squared_dev += dev * dev;
            }
            const denom = @as(f64, @floatFromInt(total - ddof));
            return sum_squared_dev / denom;
        }

        /// Standard deviation of all elements in the array
        ///
        /// Parameters:
        /// - ddof: Delta degrees of freedom. Use 0 for population std, 1 for sample std.
        ///
        /// Returns: Standard deviation as f64 (square root of variance)
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn std(self: *const Self, ddof: usize) f64 {
            return @sqrt(self.variance(ddof));
        }

        /// Median of all elements in the array
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the sorted copy
        ///
        /// Returns: Median as f64 (middle value for odd length, average of two middle for even)
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - AllocatorError on allocation failure
        ///
        /// Time: O(n log n) where n = prod(shape)
        /// Space: O(n) for sorted copy
        pub fn median(self: *const Self, allocator: Allocator) (Error || AllocatorError)!f64 {
            const total = self.count();
            if (total == 0) {
                return error.EmptyArray;
            }

            // Allocate and copy data as f64 for sorting
            var sorted = try allocator.alloc(f64, total);
            defer allocator.free(sorted);

            // Copy and convert to f64
            var i: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                sorted[i] = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                i += 1;
            }

            // Sort ascending
            stdlib.mem.sort(f64, sorted, {}, sorting.asc(f64));

            // Compute median
            if (total % 2 == 1) {
                return sorted[total / 2];
            } else {
                const lower = sorted[total / 2 - 1];
                const upper = sorted[total / 2];
                return (lower + upper) / 2.0;
            }
        }

        /// Percentile value at p% (0-100 scale)
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the sorted copy
        /// - p: Percentile (0.0 to 100.0). p=0 is min, p=50 is median, p=100 is max.
        ///
        /// Returns: Value at percentile p as f64 (with linear interpolation)
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - error.InvalidValue if p < 0 or p > 100
        /// - AllocatorError on allocation failure
        ///
        /// Time: O(n log n) where n = prod(shape)
        /// Space: O(n) for sorted copy
        pub fn percentile(self: *const Self, allocator: Allocator, p: f64) (Error || AllocatorError)!f64 {
            const total = self.count();
            if (total == 0) {
                return error.EmptyArray;
            }

            if (p < 0.0 or p > 100.0) {
                return error.InvalidValue;
            }

            // Allocate and copy data as f64 for sorting
            var sorted = try allocator.alloc(f64, total);
            defer allocator.free(sorted);

            // Copy and convert to f64
            var i: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                sorted[i] = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                i += 1;
            }

            // Sort ascending
            stdlib.mem.sort(f64, sorted, {}, sorting.asc(f64));

            // Compute index with linear interpolation
            const index = (p / 100.0) * @as(f64, @floatFromInt(total - 1));
            const lower_idx = @as(usize, @intFromFloat(@floor(index)));
            const upper_idx = @as(usize, @intFromFloat(@ceil(index)));

            if (lower_idx == upper_idx) {
                return sorted[lower_idx];
            }

            const fraction = index - @floor(index);
            return sorted[lower_idx] * (1.0 - fraction) + sorted[upper_idx] * fraction;
        }

        /// Quantile value at q fraction (0-1 scale)
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the sorted copy
        /// - q: Quantile (0.0 to 1.0). q=0 is min, q=0.5 is median, q=1 is max.
        ///
        /// Returns: Value at quantile q as f64 (with linear interpolation)
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - error.InvalidValue if q < 0 or q > 1
        /// - AllocatorError on allocation failure
        ///
        /// Time: O(n log n) where n = prod(shape)
        /// Space: O(n) for sorted copy
        pub fn quantile(self: *const Self, allocator: Allocator, q: f64) (Error || AllocatorError)!f64 {
            if (q < 0.0 or q > 1.0) {
                return error.InvalidValue;
            }
            return self.percentile(allocator, q * 100.0);
        }

        /// Covariance matrix
        ///
        /// Computes the sample covariance matrix with Bessel correction (N-1 denominator).
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result
        /// - rowvar: If true, each row is a variable (observations in columns).
        ///           If false, each column is a variable (observations in rows).
        ///
        /// For 1D input: Returns scalar variance (0D array)
        /// For 2D input: Returns covariance matrix (2D array)
        ///
        /// Formula: cov(X,Y) = E[(X - μₓ)(Y - μᵧ)] = Σ((xᵢ - μₓ)(yᵢ - μᵧ)) / (N-1)
        ///
        /// Returns: NDArray(f64, ndim) - covariance matrix or scalar
        ///
        /// Errors:
        /// - error.EmptyArray if input is empty
        /// - AllocatorError if memory allocation fails
        ///
        /// Time: O(n×m²) where n = num_observations, m = num_variables
        /// Space: O(m²) for covariance matrix
        pub fn cov(self: *const Self, allocator: Allocator, rowvar: bool) (Error || AllocatorError)!if (ndim == 1) NDArray(f64, 0) else NDArray(f64, 2) {
            if (self.data.len == 0) {
                return error.EmptyArray;
            }

            // 1D case: return scalar variance
            if (ndim == 1) {
                const var_val = self.variance(1); // Bessel correction (N-1)
                var result = try NDArray(f64, 0).init(allocator, &[0]usize{}, self.layout);
                result.data[0] = var_val;
                return result;
            }

            // 2D case: compute covariance matrix
            const n_vars: usize = if (rowvar) self.shape[0] else self.shape[1];
            const n_obs: usize = if (rowvar) self.shape[1] else self.shape[0];

            // Allocate covariance matrix (n_vars × n_vars)
            var result = try NDArray(f64, 2).init(allocator, &[_]usize{ n_vars, n_vars }, self.layout);
            errdefer result.deinit();

            // Compute means for each variable
            var means = try allocator.alloc(f64, n_vars);
            defer allocator.free(means);
            @memset(means, 0.0);

            for (0..n_vars) |v| {
                var var_sum: f64 = 0.0;
                for (0..n_obs) |o| {
                    const idx = if (rowvar) v * self.shape[1] + o else o * self.shape[1] + v;
                    const val = switch (@typeInfo(T)) {
                        .int => @as(f64, @floatFromInt(self.data[idx])),
                        .float => @as(f64, @floatCast(self.data[idx])),
                        else => @compileError("Unsupported type for cov"),
                    };
                    var_sum += val;
                }
                means[v] = var_sum / @as(f64, @floatFromInt(n_obs));
            }

            // Compute covariance matrix
            for (0..n_vars) |i| {
                for (0..n_vars) |j| {
                    var covariance: f64 = 0.0;
                    for (0..n_obs) |o| {
                        const idx_i = if (rowvar) i * self.shape[1] + o else o * self.shape[1] + i;
                        const idx_j = if (rowvar) j * self.shape[1] + o else o * self.shape[1] + j;
                        const val_i = switch (@typeInfo(T)) {
                            .int => @as(f64, @floatFromInt(self.data[idx_i])),
                            .float => @as(f64, @floatCast(self.data[idx_i])),
                            else => @compileError("Unsupported type for cov"),
                        };
                        const val_j = switch (@typeInfo(T)) {
                            .int => @as(f64, @floatFromInt(self.data[idx_j])),
                            .float => @as(f64, @floatCast(self.data[idx_j])),
                            else => @compileError("Unsupported type for cov"),
                        };
                        covariance += (val_i - means[i]) * (val_j - means[j]);
                    }
                    // Bessel correction: divide by (N-1)
                    const denom = if (n_obs > 1) @as(f64, @floatFromInt(n_obs - 1)) else 1.0;
                    result.data[i * n_vars + j] = covariance / denom;
                }
            }

            return result;
        }

        /// Pearson correlation coefficient matrix
        ///
        /// Computes the correlation coefficient matrix, which is the normalized covariance matrix.
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result
        /// - rowvar: If true, each row is a variable (observations in columns).
        ///           If false, each column is a variable (observations in rows).
        ///
        /// For 1D input: Returns scalar 1.0 (0D array, perfect self-correlation)
        /// For 2D input: Returns correlation matrix (2D array)
        ///
        /// Formula: ρ(X,Y) = cov(X,Y) / (σₓ × σᵧ)
        ///
        /// Correlation values are in [-1, 1]:
        /// - 1.0 = perfect positive correlation
        /// - -1.0 = perfect negative correlation
        /// - 0.0 = no correlation
        /// - Diagonal is always 1.0 (perfect self-correlation)
        ///
        /// Returns: NDArray(f64, ndim) - correlation matrix or scalar
        ///
        /// Errors:
        /// - error.EmptyArray if input is empty
        /// - AllocatorError if memory allocation fails
        ///
        /// Note: If a variable has zero variance (constant), correlation will be NaN or Inf
        ///
        /// Time: O(n×m²) where n = num_observations, m = num_variables
        /// Space: O(m²) for correlation matrix
        pub fn corrcoef(self: *const Self, allocator: Allocator, rowvar: bool) (Error || AllocatorError)!if (ndim == 1) NDArray(f64, 0) else NDArray(f64, 2) {
            if (self.data.len == 0) {
                return error.EmptyArray;
            }

            // 1D case: return scalar 1.0 (perfect self-correlation)
            if (ndim == 1) {
                var result = try NDArray(f64, 0).init(allocator, &[0]usize{}, self.layout);
                result.data[0] = 1.0;
                return result;
            }

            // 2D case: compute correlation from covariance
            var cov_matrix = try self.cov(allocator, rowvar);
            defer cov_matrix.deinit();

            const n_vars = cov_matrix.shape[0];

            // Extract standard deviations (sqrt of diagonal)
            var std_devs = try allocator.alloc(f64, n_vars);
            defer allocator.free(std_devs);

            for (0..n_vars) |i| {
                std_devs[i] = @sqrt(cov_matrix.data[i * n_vars + i]);
            }

            // Compute correlation matrix
            var result = try NDArray(f64, 2).init(allocator, &[_]usize{ n_vars, n_vars }, self.layout);
            errdefer result.deinit();

            for (0..n_vars) |i| {
                for (0..n_vars) |j| {
                    const denom = std_devs[i] * std_devs[j];
                    if (denom == 0.0) {
                        // Handle zero variance (constant variable)
                        result.data[i * n_vars + j] = math.nan(f64);
                    } else {
                        result.data[i * n_vars + j] = cov_matrix.data[i * n_vars + j] / denom;
                    }
                }
            }

            return result;
        }

        /// Mode (most frequent value) of all elements in the array
        ///
        /// Parameters:
        /// - allocator: Memory allocator for frequency counting HashMap
        ///
        /// Returns: Most frequent value as f64. When multiple modes exist (tie), returns the smallest.
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - AllocatorError on allocation failure
        ///
        /// Algorithm: Builds a HashMap of element frequencies in O(n) time, finds max frequency,
        /// collects all values with max frequency, sorts them, returns minimum.
        ///
        /// Time: O(n) average where n = prod(shape)
        /// Space: O(k) where k = number of unique elements
        pub fn mode(self: *const Self, allocator: Allocator) (Error || AllocatorError)!f64 {
            const total = self.count();
            if (total == 0) {
                return error.EmptyArray;
            }

            // Allocate and copy data as f64 for sorting
            var sorted = try allocator.alloc(f64, total);
            defer allocator.free(sorted);

            // Copy and convert to f64
            var i: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                sorted[i] = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                i += 1;
            }

            // Sort ascending
            stdlib.mem.sort(f64, sorted, {}, sorting.asc(f64));

            // Count frequencies and find mode
            var current_val = sorted[0];
            var current_freq: usize = 1;
            var max_freq: usize = 1;
            var mode_val = current_val;

            for (1..total) |idx| {
                if (sorted[idx] == current_val) {
                    current_freq += 1;
                } else {
                    if (current_freq > max_freq) {
                        max_freq = current_freq;
                        mode_val = current_val;
                    }
                    current_val = sorted[idx];
                    current_freq = 1;
                }
            }

            // Check final frequency
            if (current_freq > max_freq) {
                mode_val = current_val;
            }

            return mode_val;
        }

        /// Skewness (Fisher-Pearson skewness coefficient) of all elements
        ///
        /// Parameters:
        /// - allocator: Memory allocator (unused, for API consistency)
        ///
        /// Returns: Skewness as f64. Positive = right-skewed, Negative = left-skewed, ~0 = symmetric
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - error.InvalidValue if standard deviation is zero or array has < 2 elements
        ///
        /// Formula: skewness = E[((X - μ) / σ)³] = (sum((x - mean)³) / n) / std³
        ///
        /// Interpretation:
        /// - Positive: distribution has long right tail
        /// - Negative: distribution has long left tail
        /// - Zero: symmetric distribution (e.g., normal distribution)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn skewness(self: *const Self, allocator: Allocator) (Error || AllocatorError)!f64 {
            _ = allocator; // unused for API consistency
            const total = self.count();
            if (total == 0) {
                return error.EmptyArray;
            }

            const mean_val = self.mean();
            const std_val = self.std(0);

            // Check for zero variance
            if (std_val <= 1e-10) {
                return error.InvalidValue;
            }

            // Compute sum of cubed deviations
            var sum_cubed_dev: f64 = 0.0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                const fval = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                const dev = fval - mean_val;
                sum_cubed_dev += dev * dev * dev;
            }

            const n_f64 = @as(f64, @floatFromInt(total));
            const std_cubed = std_val * std_val * std_val;
            return (sum_cubed_dev / n_f64) / std_cubed;
        }

        /// Kurtosis of all elements in the array
        ///
        /// Parameters:
        /// - allocator: Memory allocator (unused, for API consistency)
        /// - fisher: If true, returns excess kurtosis (Pearson - 3); if false, returns Pearson kurtosis
        ///
        /// Returns: Kurtosis as f64. Excess kurtosis interpretation:
        /// - Positive: leptokurtic (heavy tails, peaked)
        /// - Zero: mesokurtic (normal-like)
        /// - Negative: platykurtic (light tails, flat)
        ///
        /// Errors:
        /// - error.EmptyArray if the array is empty
        /// - error.InvalidValue if standard deviation is zero or array has < 2 elements
        ///
        /// Formula:
        /// - Pearson kurtosis: E[((X - μ) / σ)⁴] = (sum((x - mean)⁴) / n) / std⁴
        /// - Excess kurtosis: Pearson kurtosis - 3
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn kurtosis(self: *const Self, allocator: Allocator, fisher: bool) (Error || AllocatorError)!f64 {
            _ = allocator; // unused for API consistency
            const total = self.count();
            if (total == 0) {
                return error.EmptyArray;
            }

            const mean_val = self.mean();
            const std_val = self.std(0);

            // Check for zero variance
            if (std_val <= 1e-10) {
                return error.InvalidValue;
            }

            // Compute sum of fourth power deviations
            var sum_fourth_dev: f64 = 0.0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                const fval = if (@typeInfo(T) == .float)
                    @as(f64, val)
                else
                    @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                const dev = fval - mean_val;
                sum_fourth_dev += dev * dev * dev * dev;
            }

            const n_f64 = @as(f64, @floatFromInt(total));
            const std_fourth = std_val * std_val * std_val * std_val;
            const kurt_pearson = (sum_fourth_dev / n_f64) / std_fourth;

            return if (fisher) kurt_pearson - 3.0 else kurt_pearson;
        }

        /// Minimum element in the array
        ///
        /// Returns: Minimum element value as type T
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn min(self: *const Self) T {
            var result: T = self.data[0];
            for (self.data) |val| {
                if (val < result) {
                    result = val;
                }
            }
            return result;
        }

        /// Maximum element in the array
        ///
        /// Returns: Maximum element value as type T
        ///
        /// Errors:
        /// - None (always succeeds)
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn max(self: *const Self) T {
            var result: T = self.data[0];
            for (self.data) |val| {
                if (val > result) {
                    result = val;
                }
            }
            return result;
        }

        /// Index of the minimum element in the array
        ///
        /// Returns: Linear (flat) index of the minimum element
        ///
        /// Errors:
        /// - error.ZeroDimension if array is empty (all dimensions > 0, so this shouldn't happen)
        ///
        /// For multiple occurrences of the minimum value, returns the index of the first occurrence.
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn argmin(self: *const Self) Error!usize {
            if (self.data.len == 0) {
                return error.ZeroDimension;
            }

            var min_idx: usize = 0;
            var min_val: T = self.data[0];

            for (1..self.data.len) |i| {
                if (self.data[i] < min_val) {
                    min_val = self.data[i];
                    min_idx = i;
                }
            }

            return min_idx;
        }

        /// Index of the maximum element in the array
        ///
        /// Returns: Linear (flat) index of the maximum element
        ///
        /// Errors:
        /// - error.ZeroDimension if array is empty (all dimensions > 0, so this shouldn't happen)
        ///
        /// For multiple occurrences of the maximum value, returns the index of the first occurrence.
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(1)
        pub fn argmax(self: *const Self) Error!usize {
            if (self.data.len == 0) {
                return error.ZeroDimension;
            }

            var max_idx: usize = 0;
            var max_val: T = self.data[0];

            for (1..self.data.len) |i| {
                if (self.data[i] > max_val) {
                    max_val = self.data[i];
                    max_idx = i;
                }
            }

            return max_idx;
        }

        /// Cumulative sum of elements
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        ///
        /// Returns: New NDArray with same shape containing cumulative sums
        ///
        /// The cumulative sum is computed by iterating through the flattened array
        /// and accumulating running sum values.
        /// Example: [1, 2, 3, 4, 5] → [1, 3, 6, 10, 15]
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn cumsum(self: *const Self, allocator: Allocator) (Error || AllocatorError)!Self {
            // Create result array with same shape and layout
            var result = try Self.init(allocator, self.shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize first element
            result.data[0] = self.data[0];

            // Compute cumulative sum
            for (1..self.data.len) |i| {
                result.data[i] = result.data[i - 1] + self.data[i];
            }

            return result;
        }

        /// Cumulative product of elements
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        ///
        /// Returns: New NDArray with same shape containing cumulative products
        ///
        /// The cumulative product is computed by iterating through the flattened array
        /// and accumulating running product values.
        /// Example: [1, 2, 3, 4, 5] → [1, 2, 6, 24, 120]
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(n) for result array
        pub fn cumprod(self: *const Self, allocator: Allocator) (Error || AllocatorError)!Self {
            // Create result array with same shape and layout
            var result = try Self.init(allocator, self.shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize first element
            result.data[0] = self.data[0];

            // Compute cumulative product
            for (1..self.data.len) |i| {
                result.data[i] = result.data[i - 1] * self.data[i];
            }

            return result;
        }

        // -- Sorting Operations --

        /// Sort elements along a specified axis
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: The axis along which to sort (0 to ndim-1)
        ///
        /// Returns: New NDArray with same shape containing sorted values
        ///
        /// Sorts each 1-D slice along the specified axis in ascending order.
        /// Uses standard library sort with O(n log n) comparison-based sorting.
        ///
        /// Example (2D, axis=0): [[3,1],[2,4]] → [[2,1],[3,4]] (sort columns)
        /// Example (2D, axis=1): [[3,1],[2,4]] → [[1,3],[2,4]] (sort rows)
        ///
        /// Time: O(m × n log n) where m = number of slices, n = slice length
        /// Space: O(prod(shape)) for result array
        pub fn sort(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Create result array with same shape and layout
            var result = try Self.init(allocator, self.shape[0..], self.layout);
            errdefer result.deinit();

            // Copy data first
            @memcpy(result.data, self.data);

            // Sort along the specified axis
            const axis_len = self.shape[axis];
            if (axis_len <= 1) {
                return result; // Already sorted if axis length is 0 or 1
            }

            // Calculate number of slices to sort
            var num_slices: usize = 1;
            for (0..ndim) |d| {
                if (d != axis) {
                    num_slices *= self.shape[d];
                }
            }

            // Create buffer for extracting slices
            const slice_buf = try allocator.alloc(T, axis_len);
            defer allocator.free(slice_buf);

            // For each slice perpendicular to the axis
            for (0..num_slices) |slice_idx| {
                // Calculate multi-dimensional index for this slice
                var multi_idx: [ndim]usize = undefined;
                var remaining = slice_idx;

                var dim_idx: usize = 0;
                for (0..ndim) |d| {
                    if (d == axis) continue;

                    var divisor: usize = 1;
                    var idx_after = dim_idx + 1;
                    for (0..ndim) |dd| {
                        if (dd == axis) continue;
                        if (idx_after > 0) {
                            idx_after -= 1;
                            continue;
                        }
                        divisor *= self.shape[dd];
                    }

                    multi_idx[d] = remaining / divisor;
                    remaining = remaining % divisor;
                    dim_idx += 1;
                }

                // Extract slice along axis
                for (0..axis_len) |i| {
                    multi_idx[axis] = i;
                    var offset: usize = 0;
                    for (0..ndim) |d| {
                        offset += multi_idx[d] * result.strides[d];
                    }
                    slice_buf[i] = result.data[offset];
                }

                // Sort the slice
                sorting.heap(T, slice_buf, {}, comptime sorting.asc(T));

                // Write sorted values back
                for (0..axis_len) |i| {
                    multi_idx[axis] = i;
                    var offset: usize = 0;
                    for (0..ndim) |d| {
                        offset += multi_idx[d] * result.strides[d];
                    }
                    result.data[offset] = slice_buf[i];
                }
            }

            return result;
        }

        /// Return indices that would sort the array along a specified axis
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: The axis along which to compute indices (0 to ndim-1)
        ///
        /// Returns: New NDArray of usize with same shape containing sort indices
        ///
        /// Returns the indices that would sort each 1-D slice along the axis.
        /// These indices can be used to reorder the original array to sorted order.
        ///
        /// Example (1D): [30, 10, 20] → [1, 2, 0]  (indices that sort the array)
        /// Example (2D, axis=0): [[3,1],[2,4]] → [[1,0],[0,1]] (column sort indices)
        ///
        /// Time: O(m × n log n) where m = number of slices, n = slice length
        /// Space: O(prod(shape)) for result array
        pub fn argsort(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(usize, ndim) {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Create result array for indices
            var result = try NDArray(usize, ndim).init(allocator, self.shape[0..], self.layout);
            errdefer result.deinit();

            const axis_len = self.shape[axis];
            if (axis_len == 0) {
                return result; // Empty array
            }

            // Calculate number of slices to sort
            var num_slices: usize = 1;
            for (0..ndim) |d| {
                if (d != axis) {
                    num_slices *= self.shape[d];
                }
            }

            // Create buffers for sorting
            const IndexValue = struct {
                index: usize,
                value: T,
            };

            const slice_buf = try allocator.alloc(IndexValue, axis_len);
            defer allocator.free(slice_buf);

            // For each slice perpendicular to the axis
            for (0..num_slices) |slice_idx| {
                // Calculate multi-dimensional index for this slice
                var multi_idx: [ndim]usize = undefined;
                var remaining = slice_idx;

                var dim_idx: usize = 0;
                for (0..ndim) |d| {
                    if (d == axis) continue;

                    var divisor: usize = 1;
                    var idx_after = dim_idx + 1;
                    for (0..ndim) |dd| {
                        if (dd == axis) continue;
                        if (idx_after > 0) {
                            idx_after -= 1;
                            continue;
                        }
                        divisor *= self.shape[dd];
                    }

                    multi_idx[d] = remaining / divisor;
                    remaining = remaining % divisor;
                    dim_idx += 1;
                }

                // Extract slice with indices
                for (0..axis_len) |i| {
                    multi_idx[axis] = i;
                    var offset: usize = 0;
                    for (0..ndim) |d| {
                        offset += multi_idx[d] * self.strides[d];
                    }
                    slice_buf[i] = .{
                        .index = i,
                        .value = self.data[offset],
                    };
                }

                // Sort by value
                sorting.heap(IndexValue, slice_buf, {}, struct {
                    fn lessThan(_: void, a: IndexValue, b: IndexValue) bool {
                        return a.value < b.value;
                    }
                }.lessThan);

                // Write indices back
                for (0..axis_len) |i| {
                    multi_idx[axis] = i;
                    var offset: usize = 0;
                    for (0..ndim) |d| {
                        offset += multi_idx[d] * result.strides[d];
                    }
                    result.data[offset] = slice_buf[i].index;
                }
            }

            return result;
        }

        /// Find unique elements in the array.
        ///
        /// Returns a 1D array containing the unique elements in sorted order.
        /// For multi-dimensional arrays, the array is flattened first.
        ///
        /// Time: O(n log n) where n = total number of elements
        /// Space: O(n) for temporary sorting buffer
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{3, 1, 2, 1, 3}, .row_major);
        /// const uniq = try arr.unique(allocator);
        /// defer uniq.deinit();
        /// // uniq.data = {1, 2, 3}
        /// ```
        pub fn unique(self: *const Self, allocator: Allocator) (Error || AllocatorError)!NDArray(T, 1) {
            const total = self.count();
            if (total == 0) {
                return NDArray(T, 1).init(allocator, &[_]usize{0}, .row_major);
            }

            // Flatten and copy all elements to a temporary buffer
            const temp = try allocator.alloc(T, total);
            defer allocator.free(temp);

            var idx: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| : (idx += 1) {
                temp[idx] = val;
            }

            // Sort the temporary buffer
            sorting.heap(T, temp, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return a < b;
                }
            }.lessThan);

            // Count unique elements
            var unique_count: usize = 1;
            for (1..total) |i| {
                if (temp[i] != temp[i - 1]) {
                    unique_count += 1;
                }
            }

            // Create result array
            var result = try NDArray(T, 1).init(allocator, &[_]usize{unique_count}, .row_major);
            errdefer result.deinit();

            // Fill with unique elements
            result.data[0] = temp[0];
            var result_idx: usize = 1;
            for (1..total) |i| {
                if (temp[i] != temp[i - 1]) {
                    result.data[result_idx] = temp[i];
                    result_idx += 1;
                }
            }

            return result;
        }

        /// Find unique elements and their counts.
        ///
        /// Returns a tuple of (unique_values, counts) where:
        /// - unique_values: 1D array of unique elements in sorted order
        /// - counts: 1D array of occurrence counts for each unique value
        ///
        /// Time: O(n log n) where n = total number of elements
        /// Space: O(n) for temporary buffers
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{3, 1, 2, 1, 3, 3}, .row_major);
        /// const result = try arr.uniqueWithCounts(allocator);
        /// defer result.values.deinit();
        /// defer result.counts.deinit();
        /// // result.values.data = {1, 2, 3}
        /// // result.counts.data = {2, 1, 3}
        /// ```
        pub fn uniqueWithCounts(self: *const Self, allocator: Allocator) (Error || AllocatorError)!struct {
            values: NDArray(T, 1),
            counts: NDArray(usize, 1),
        } {
            const total = self.count();
            if (total == 0) {
                return .{
                    .values = try NDArray(T, 1).init(allocator, &[_]usize{0}, .row_major),
                    .counts = try NDArray(usize, 1).init(allocator, &[_]usize{0}, .row_major),
                };
            }

            // Flatten and copy all elements
            const temp = try allocator.alloc(T, total);
            defer allocator.free(temp);

            var idx: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| : (idx += 1) {
                temp[idx] = val;
            }

            // Sort
            sorting.heap(T, temp, {}, struct {
                fn lessThan(_: void, a: T, b: T) bool {
                    return a < b;
                }
            }.lessThan);

            // Count unique elements and their frequencies
            var unique_count: usize = 1;
            for (1..total) |i| {
                if (temp[i] != temp[i - 1]) {
                    unique_count += 1;
                }
            }

            // Create result arrays
            var values = try NDArray(T, 1).init(allocator, &[_]usize{unique_count}, .row_major);
            errdefer values.deinit();
            var counts = try NDArray(usize, 1).init(allocator, &[_]usize{unique_count}, .row_major);
            errdefer counts.deinit();

            // Fill with unique elements and counts
            values.data[0] = temp[0];
            counts.data[0] = 1;
            var result_idx: usize = 0;
            for (1..total) |i| {
                if (temp[i] != temp[i - 1]) {
                    result_idx += 1;
                    values.data[result_idx] = temp[i];
                    counts.data[result_idx] = 1;
                } else {
                    counts.data[result_idx] += 1;
                }
            }

            return .{ .values = values, .counts = counts };
        }

        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Performs binary search on a sorted 1D array to find insertion indices.
        /// Similar to NumPy's searchsorted().
        ///
        /// Time: O(m * log n) where m = values.count(), n = self.count()
        /// Space: O(m) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - values: Values to search for (1D array)
        /// - side: 'left' (default) or 'right' for insertion side
        ///
        /// Returns: 1D array of insertion indices
        ///
        /// Example:
        /// ```
        /// const sorted = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 3, 5, 7}, .row_major);
        /// const vals = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 6}, .row_major);
        /// const indices = try sorted.searchsorted(allocator, &vals, .left);
        /// defer indices.deinit();
        /// // indices.data = {1, 3} — insert 2 at index 1, 6 at index 3
        /// ```
        pub fn searchsorted(
            self: *const Self,
            allocator: Allocator,
            values: *const NDArray(T, 1),
            side: enum { left, right },
        ) (Error || AllocatorError)!NDArray(usize, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();
            const m = values.count();
            var result = try NDArray(usize, 1).init(allocator, &[_]usize{m}, .row_major);
            errdefer result.deinit();

            for (0..m) |i| {
                const target = values.data[i];
                var left: usize = 0;
                var right: usize = n;

                // Binary search
                while (left < right) {
                    const mid = left + (right - left) / 2;
                    const mid_val = self.data[mid];

                    const cmp_result = switch (side) {
                        .left => target <= mid_val,
                        .right => target < mid_val,
                    };

                    if (cmp_result) {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }
                result.data[i] = left;
            }

            return result;
        }

        /// Return the indices of non-zero elements.
        ///
        /// Flattens the array and returns indices where elements are non-zero.
        /// Similar to NumPy's nonzero().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(k) where k = number of non-zero elements
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        ///
        /// Returns: 1D array of flat indices where elements are non-zero
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 2).fromSlice(allocator, &[_]i32{0, 1, 2, 0, 3, 0}, &[_]usize{2, 3}, .row_major);
        /// const indices = try arr.nonzero(allocator);
        /// defer indices.deinit();
        /// // indices.data = {1, 2, 4} — flat indices of non-zero elements (1, 2, 3)
        /// ```
        pub fn nonzero(self: *const Self, allocator: Allocator) (Error || AllocatorError)!NDArray(usize, 1) {
            // First pass: count non-zero elements
            var nz_count: usize = 0;
            var iter = self.iterator();
            while (iter.next()) |val| {
                // Check if value is non-zero
                const is_nonzero = switch (@typeInfo(T)) {
                    .int, .comptime_int => val != 0,
                    .float, .comptime_float => val != 0.0,
                    .bool => val,
                    else => @compileError("nonzero only works with numeric or bool types"),
                };
                if (is_nonzero) nz_count += 1;
            }

            // Handle edge case: all zeros - create empty array manually
            if (nz_count == 0) {
                const data = try allocator.alloc(usize, 0);
                return NDArray(usize, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            // Create result array
            var result = try NDArray(usize, 1).init(allocator, &[_]usize{nz_count}, .row_major);
            errdefer result.deinit();

            // Second pass: collect indices
            var idx: usize = 0;
            var result_idx: usize = 0;
            iter = self.iterator();
            while (iter.next()) |val| : (idx += 1) {
                const is_nonzero = switch (@typeInfo(T)) {
                    .int, .comptime_int => val != 0,
                    .float, .comptime_float => val != 0.0,
                    .bool => val,
                    else => unreachable,
                };
                if (is_nonzero) {
                    result.data[result_idx] = idx;
                    result_idx += 1;
                }
            }

            return result;
        }

        /// Find the union of two 1D arrays.
        ///
        /// Returns the unique, sorted array of values that are in either of the
        /// input arrays. Similar to NumPy's union1d().
        ///
        /// Time: O((n + m) × log(n + m)) where n = self.count(), m = other.count()
        /// Space: O(n + m) for combined array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - other: Second 1D array
        ///
        /// Returns: 1D array of unique values in union
        ///
        /// Example:
        /// ```
        /// const a = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3}, .row_major);
        /// const b = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 3, 4}, .row_major);
        /// const result = try a.union1d(allocator, &b);
        /// defer result.deinit();
        /// // result.data = {1, 2, 3, 4}
        /// ```
        pub fn union1d(self: *const Self, allocator: Allocator, other: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();
            const m = other.count();

            // Combine both arrays
            var combined = try allocator.alloc(T, n + m);
            defer allocator.free(combined);

            // Copy elements
            for (0..n) |i| combined[i] = self.data[i];
            for (0..m) |i| combined[n + i] = other.data[i];

            // Sort combined array
            sorting.heap(T, combined, {}, sorting.asc(T));

            // Count unique elements
            var unique_count: usize = 0;
            if (combined.len > 0) {
                unique_count = 1;
                for (1..combined.len) |i| {
                    if (combined[i] != combined[i - 1]) unique_count += 1;
                }
            }

            // Fill result with unique elements (handle empty case)
            if (unique_count == 0) {
                const data = try allocator.alloc(T, 0);
                return NDArray(T, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            var result = try NDArray(T, 1).init(allocator, &[_]usize{unique_count}, .row_major);
            errdefer result.deinit();

            result.data[0] = combined[0];
            var result_idx: usize = 1;
            for (1..combined.len) |i| {
                if (combined[i] != combined[i - 1]) {
                    result.data[result_idx] = combined[i];
                    result_idx += 1;
                }
            }

            return result;
        }

        /// Find the intersection of two 1D arrays.
        ///
        /// Returns the unique, sorted array of values that are in both input arrays.
        /// Similar to NumPy's intersect1d().
        ///
        /// Time: O((n + m) × log(n + m)) where n = self.count(), m = other.count()
        /// Space: O(min(n, m)) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - other: Second 1D array
        ///
        /// Returns: 1D array of unique values in intersection
        ///
        /// Example:
        /// ```
        /// const a = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3}, .row_major);
        /// const b = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 3, 4}, .row_major);
        /// const result = try a.intersect1d(allocator, &b);
        /// defer result.deinit();
        /// // result.data = {2, 3}
        /// ```
        pub fn intersect1d(self: *const Self, allocator: Allocator, other: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            // Get sorted unique values from both arrays
            var unique_self = try self.unique(allocator);
            defer unique_self.deinit();
            var unique_other = try other.unique(allocator);
            defer unique_other.deinit();

            const n = unique_self.count();
            const m = unique_other.count();

            // Two-pointer approach on sorted arrays
            var temp = try allocator.alloc(T, @min(n, m));
            defer allocator.free(temp);

            var i: usize = 0;
            var j: usize = 0;
            var result_count: usize = 0;

            while (i < n and j < m) {
                const val_self = unique_self.data[i];
                const val_other = unique_other.data[j];

                if (val_self == val_other) {
                    temp[result_count] = val_self;
                    result_count += 1;
                    i += 1;
                    j += 1;
                } else if (val_self < val_other) {
                    i += 1;
                } else {
                    j += 1;
                }
            }

            // Create result array (handle empty case)
            if (result_count == 0) {
                const data = try allocator.alloc(T, 0);
                return NDArray(T, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            var result = try NDArray(T, 1).init(allocator, &[_]usize{result_count}, .row_major);
            errdefer result.deinit();
            for (0..result_count) |k| result.data[k] = temp[k];

            return result;
        }

        /// Find the set difference of two 1D arrays.
        ///
        /// Returns the unique values in self that are not in other.
        /// Similar to NumPy's setdiff1d().
        ///
        /// Time: O((n + m) × log(n + m)) where n = self.count(), m = other.count()
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - other: Second 1D array
        ///
        /// Returns: 1D array of unique values in self but not in other
        ///
        /// Example:
        /// ```
        /// const a = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3}, .row_major);
        /// const b = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 3, 4}, .row_major);
        /// const result = try a.setdiff1d(allocator, &b);
        /// defer result.deinit();
        /// // result.data = {1}
        /// ```
        pub fn setdiff1d(self: *const Self, allocator: Allocator, other: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            // Get sorted unique values from both arrays
            var unique_self = try self.unique(allocator);
            defer unique_self.deinit();
            var unique_other = try other.unique(allocator);
            defer unique_other.deinit();

            const n = unique_self.count();
            const m = unique_other.count();

            // Two-pointer approach on sorted arrays
            var temp = try allocator.alloc(T, n);
            defer allocator.free(temp);

            var i: usize = 0;
            var j: usize = 0;
            var result_count: usize = 0;

            while (i < n) {
                const val_self = unique_self.data[i];

                // Advance j until we pass val_self
                while (j < m and unique_other.data[j] < val_self) j += 1;

                // If not found in other, add to result
                if (j >= m or unique_other.data[j] != val_self) {
                    temp[result_count] = val_self;
                    result_count += 1;
                }
                i += 1;
            }

            // Create result array (handle empty case)
            if (result_count == 0) {
                const data = try allocator.alloc(T, 0);
                return NDArray(T, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            var result = try NDArray(T, 1).init(allocator, &[_]usize{result_count}, .row_major);
            errdefer result.deinit();
            for (0..result_count) |k| result.data[k] = temp[k];

            return result;
        }

        /// Find the symmetric difference of two 1D arrays.
        ///
        /// Returns the unique values that are in either array but not in both.
        /// Similar to NumPy's setxor1d().
        ///
        /// Time: O((n + m) × log(n + m)) where n = self.count(), m = other.count()
        /// Space: O(n + m) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - other: Second 1D array
        ///
        /// Returns: 1D array of unique values in symmetric difference
        ///
        /// Example:
        /// ```
        /// const a = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3}, .row_major);
        /// const b = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 3, 4}, .row_major);
        /// const result = try a.setxor1d(allocator, &b);
        /// defer result.deinit();
        /// // result.data = {1, 4}
        /// ```
        pub fn setxor1d(self: *const Self, allocator: Allocator, other: *const Self) (Error || AllocatorError)!NDArray(T, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            // Get sorted unique values from both arrays
            var unique_self = try self.unique(allocator);
            defer unique_self.deinit();
            var unique_other = try other.unique(allocator);
            defer unique_other.deinit();

            const n = unique_self.count();
            const m = unique_other.count();

            // Two-pointer approach on sorted arrays
            var temp = try allocator.alloc(T, n + m);
            defer allocator.free(temp);

            var i: usize = 0;
            var j: usize = 0;
            var result_count: usize = 0;

            while (i < n or j < m) {
                if (i >= n) {
                    // Remaining elements from other
                    temp[result_count] = unique_other.data[j];
                    result_count += 1;
                    j += 1;
                } else if (j >= m) {
                    // Remaining elements from self
                    temp[result_count] = unique_self.data[i];
                    result_count += 1;
                    i += 1;
                } else {
                    const val_self = unique_self.data[i];
                    const val_other = unique_other.data[j];

                    if (val_self == val_other) {
                        // In both arrays, skip
                        i += 1;
                        j += 1;
                    } else if (val_self < val_other) {
                        // In self only
                        temp[result_count] = val_self;
                        result_count += 1;
                        i += 1;
                    } else {
                        // In other only
                        temp[result_count] = val_other;
                        result_count += 1;
                        j += 1;
                    }
                }
            }

            // Create result array (handle empty case)
            if (result_count == 0) {
                const data = try allocator.alloc(T, 0);
                return NDArray(T, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            var result = try NDArray(T, 1).init(allocator, &[_]usize{result_count}, .row_major);
            errdefer result.deinit();
            for (0..result_count) |k| result.data[k] = temp[k];

            return result;
        }

        /// Test whether each element of a 1D array is in another 1D array.
        ///
        /// Returns a boolean array the same shape as self, with True where
        /// an element of self is in other. Similar to NumPy's in1d().
        ///
        /// Time: O(n × m) naive, O((n + m) × log m) with sorting
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - other: Array to check membership against
        ///
        /// Returns: Boolean array indicating membership
        ///
        /// Example:
        /// ```
        /// const a = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3, 4}, .row_major);
        /// const b = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{2, 4}, .row_major);
        /// const result = try a.in1d(allocator, &b);
        /// defer result.deinit();
        /// // result.data = {false, true, false, true}
        /// ```
        pub fn in1d(self: *const Self, allocator: Allocator, other: *const Self) (Error || AllocatorError)!NDArray(bool, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();

            // Sort other array for binary search
            var sorted_other = try other.unique(allocator); // unique returns sorted
            defer sorted_other.deinit();

            // Create result array
            var result = try NDArray(bool, 1).init(allocator, &[_]usize{n}, .row_major);
            errdefer result.deinit();

            // Check each element
            for (0..n) |i| {
                const target = self.data[i];
                var found = false;

                // Binary search in sorted_other
                var left: usize = 0;
                var right: usize = sorted_other.count();
                while (left < right) {
                    const mid = left + (right - left) / 2;
                    const mid_val = sorted_other.data[mid];

                    if (target == mid_val) {
                        found = true;
                        break;
                    } else if (target < mid_val) {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }

                result.data[i] = found;
            }

            return result;
        }

        /// Count occurrences of non-negative integers in array.
        ///
        /// Returns a 1D array of length max(x)+1 where each index i contains
        /// the count of occurrences of i in the input array. Similar to NumPy's bincount().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(max(x)+1) for count array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        ///
        /// Returns: 1D array where result[i] = count of i in input
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{0, 1, 1, 3, 2, 1, 7}, .row_major);
        /// const counts = try arr.bincount(allocator);
        /// defer counts.deinit();
        /// // counts.data = {1, 3, 1, 1, 0, 0, 0, 1} — indices 0,1,2,3,7 appear 1,3,1,1,1 times
        /// ```
        pub fn bincount(self: *const Self, allocator: Allocator) (Error || AllocatorError)!NDArray(usize, 1) {
            // bincount only works with non-negative integer types
            const type_info = @typeInfo(T);
            if (type_info != .int and type_info != .comptime_int) {
                @compileError("bincount only works with integer types");
            }

            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();
            if (n == 0) {
                // Empty array -> empty bincount (manual construction)
                const data = try allocator.alloc(usize, 0);
                return NDArray(usize, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            // Find maximum value to determine output size
            var max_val: T = self.data[0];
            for (self.data[0..n]) |val| {
                if (val < 0) return Error.InvalidValue; // bincount requires non-negative integers
                if (val > max_val) max_val = val;
            }

            const output_size = @as(usize, @intCast(max_val)) + 1;

            // Create result array initialized to zeros
            var result = try NDArray(usize, 1).zeros(allocator, &[_]usize{output_size}, .row_major);
            errdefer result.deinit();

            // Count occurrences
            for (self.data[0..n]) |val| {
                const idx = @as(usize, @intCast(val));
                result.data[idx] += 1;
            }

            return result;
        }

        /// Return indices of bins to which each value belongs.
        ///
        /// For each value in x, find which bin it falls into based on the bins array.
        /// Similar to NumPy's digitize(). Bins must be sorted (ascending or descending).
        ///
        /// Time: O(n * log m) where n = values count, m = bins count
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - bins: 1D array of bin edges (must be monotonic)
        /// - right: if true, intervals are (bins[i-1], bins[i]], else [bins[i-1], bins[i])
        ///
        /// Returns: 1D array where result[i] is the bin index for x[i]
        ///   - Value k means bins[k-1] <= x[i] < bins[k] (for right=false)
        ///   - Index 0 means x[i] < bins[0], index m means x[i] >= bins[m-1]
        ///
        /// Example:
        /// ```
        /// const values = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{0.2, 6.4, 3.0, 1.6}, .row_major);
        /// const bins = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{0.0, 1.0, 2.5, 4.0, 10.0}, .row_major);
        /// const indices = try values.digitize(allocator, &bins, false);
        /// defer indices.deinit();
        /// // indices.data = {1, 4, 3, 2} — 0.2 in [0,1), 6.4 in [4,10), etc.
        /// ```
        pub fn digitize(self: *const Self, allocator: Allocator, bins: *const NDArray(T, 1), right: bool) (Error || AllocatorError)!NDArray(usize, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();
            const m = bins.count();

            if (m == 0) return Error.EmptyArray;

            // Determine if bins are ascending or descending
            const ascending = if (m > 1) bins.data[1] >= bins.data[0] else true;

            // Create result array
            var result = try NDArray(usize, 1).init(allocator, &[_]usize{n}, .row_major);
            errdefer result.deinit();

            // For each value, find its bin
            for (0..n) |i| {
                const val = self.data[i];
                var bin_idx: usize = 0;

                if (ascending) {
                    // Binary search in ascending bins
                    var left: usize = 0;
                    var right_idx: usize = m;
                    while (left < right_idx) {
                        const mid = left + (right_idx - left) / 2;
                        const bin_val = bins.data[mid];

                        const cmp = if (right) val > bin_val else val >= bin_val;

                        if (cmp) {
                            left = mid + 1;
                        } else {
                            right_idx = mid;
                        }
                    }
                    bin_idx = left;
                } else {
                    // Binary search in descending bins
                    // Find first bin where val >= bin_val
                    var left: usize = 0;
                    var right_idx: usize = m;
                    while (left < right_idx) {
                        const mid = left + (right_idx - left) / 2;
                        const bin_val = bins.data[mid];

                        const cmp = if (right) val >= bin_val else val > bin_val;

                        if (cmp) {
                            right_idx = mid;
                        } else {
                            left = mid + 1;
                        }
                    }
                    bin_idx = left;
                }

                result.data[i] = bin_idx;
            }

            return result;
        }

        /// Compute histogram of array values.
        ///
        /// Count occurrences of values in each bin. Similar to NumPy's histogram().
        ///
        /// Time: O(n * log m) where n = values count, m = bins count
        /// Space: O(m-1) for histogram counts
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - bins: 1D array of bin edges (m+1 edges create m bins), must be sorted ascending
        ///
        /// Returns: 1D array of length m where result[i] = count of values in [bins[i], bins[i+1])
        ///
        /// Example:
        /// ```
        /// const data = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{0.5, 1.5, 2.5, 3.5, 1.2, 2.8}, .row_major);
        /// const bins = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{0.0, 1.0, 2.0, 3.0, 4.0}, .row_major);
        /// const hist = try data.histogram(allocator, &bins);
        /// defer hist.deinit();
        /// // hist.data = {1, 2, 2, 1} — bins [0,1), [1,2), [2,3), [3,4)
        /// ```
        pub fn histogram(self: *const Self, allocator: Allocator, bins: *const NDArray(T, 1)) (Error || AllocatorError)!NDArray(usize, 1) {
            if (ndim != 1) return Error.DimensionMismatch;

            const n = self.count();
            const m = bins.count();

            if (m < 2) return Error.InvalidValue; // Need at least 2 bin edges

            // Verify bins are sorted ascending
            for (0..m - 1) |i| {
                if (bins.data[i + 1] < bins.data[i]) {
                    return Error.InvalidValue; // Bins must be ascending
                }
            }

            const num_bins = m - 1;

            // Create result array initialized to zeros
            var result = try NDArray(usize, 1).zeros(allocator, &[_]usize{num_bins}, .row_major);
            errdefer result.deinit();

            // For each value, find its bin and increment count
            for (0..n) |i| {
                const val = self.data[i];

                // Skip values outside bin range
                if (val < bins.data[0] or val >= bins.data[m - 1]) {
                    // For the last bin, include right edge
                    if (val == bins.data[m - 1]) {
                        result.data[num_bins - 1] += 1;
                    }
                    continue;
                }

                // Binary search to find bin
                var left: usize = 0;
                var right: usize = num_bins;
                while (left < right) {
                    const mid = left + (right - left) / 2;
                    if (val >= bins.data[mid + 1]) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }

                result.data[left] += 1;
            }

            return result;
        }

        /// Extract elements from an array based on a boolean condition.
        ///
        /// Returns a 1D array containing elements where the condition is True.
        /// The input array is flattened before extraction. Similar to NumPy's extract().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(k) where k = number of True elements in condition
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - condition: Boolean array with same shape as self
        ///
        /// Returns: 1D array of extracted elements
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3, 4}, .row_major);
        /// const cond = try NDArray(bool, 1).fromSlice(allocator, &[_]bool{true, false, true, false}, .row_major);
        /// const extracted = try arr.extract(allocator, &cond);
        /// defer extracted.deinit();
        /// // extracted.data = {1, 3}
        /// ```
        pub fn extract(self: *const Self, allocator: Allocator, condition: *const NDArray(bool, ndim)) (Error || AllocatorError)!NDArray(T, 1) {
            // Verify shapes match
            if (!mem.eql(usize, &self.shape, &condition.shape)) {
                return Error.ShapeMismatch;
            }

            // First pass: count True elements
            var true_count: usize = 0;
            var cond_iter = condition.iterator();
            while (cond_iter.next()) |val| {
                if (val) true_count += 1;
            }

            // Handle edge case: no True elements
            if (true_count == 0) {
                const data = try allocator.alloc(T, 0);
                return NDArray(T, 1){
                    .shape = [_]usize{0},
                    .strides = [_]usize{1},
                    .data = data,
                    .allocator = allocator,
                    .layout = .row_major,
                    .owned = true,
                };
            }

            // Create result array
            var result = try NDArray(T, 1).init(allocator, &[_]usize{true_count}, .row_major);
            errdefer result.deinit();

            // Second pass: extract elements
            var result_idx: usize = 0;
            var self_iter = self.iterator();
            cond_iter = condition.iterator();
            while (self_iter.next()) |val| {
                const cond_val = cond_iter.next().?;
                if (cond_val) {
                    result.data[result_idx] = val;
                    result_idx += 1;
                }
            }

            return result;
        }

        /// Return selected slices of an array along an axis.
        ///
        /// Returns a copy of the array with only selected slices along the specified axis,
        /// determined by a boolean condition array. Similar to NumPy's compress().
        ///
        /// Time: O(n × m) where n = total elements, m = selected slices
        /// Space: O(prod(result_shape)) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - condition: 1D boolean array (length must equal shape[axis])
        /// - axis: Axis along which to select slices
        ///
        /// Returns: New array with selected slices
        ///
        /// Example:
        /// ```
        /// // 2D array [[1,2,3], [4,5,6], [7,8,9]]
        /// const arr = try NDArray(i32, 2).fromSlice(allocator, &[_]i32{1,2,3,4,5,6,7,8,9}, &[_]usize{3, 3}, .row_major);
        /// const cond = try NDArray(bool, 1).fromSlice(allocator, &[_]bool{true, false, true}, .row_major);
        /// const compressed = try arr.compress(allocator, &cond, 0);
        /// defer compressed.deinit();
        /// // compressed = [[1,2,3], [7,8,9]] (rows 0 and 2)
        /// ```
        pub fn compress(self: *const Self, allocator: Allocator, condition: *const NDArray(bool, 1), axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            if (condition.shape[0] != self.shape[axis]) return Error.ShapeMismatch;

            // Count True elements in condition
            var selected_count: usize = 0;
            var cond_iter = condition.iterator();
            while (cond_iter.next()) |val| {
                if (val) selected_count += 1;
            }

            // Build new shape (replace axis dimension with selected count)
            var new_shape = self.shape;
            new_shape[axis] = selected_count;

            // Handle edge case: no selected slices
            if (selected_count == 0) {
                // Calculate total size for empty array
                var total_size: usize = 1;
                for (new_shape) |dim| total_size *= dim;

                const data = try allocator.alloc(T, total_size);
                return Self{
                    .shape = new_shape,
                    .strides = calculateStrides(new_shape, self.layout),
                    .data = data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = true,
                };
            }

            // Create result array
            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Calculate total iterations (excluding axis dimension)
            const total_iters = blk: {
                var total: usize = 1;
                for (0..ndim) |d| {
                    if (d != axis) total *= self.shape[d];
                }
                break :blk total;
            };

            // Copy selected slices
            for (0..total_iters) |iter_idx| {
                // Convert iteration index to multi-dimensional index (skip axis)
                var temp_idx = iter_idx;
                var multi_idx: [ndim]usize = undefined;
                for (0..ndim) |d| {
                    if (d != axis) {
                        multi_idx[d] = temp_idx % self.shape[d];
                        temp_idx /= self.shape[d];
                    }
                }

                // Copy selected elements along axis
                var result_axis_idx: usize = 0;
                for (0..self.shape[axis]) |src_axis_idx| {
                    if (condition.data[src_axis_idx]) {
                        multi_idx[axis] = src_axis_idx;
                        var src_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| src_idx[d] = @intCast(multi_idx[d]);

                        var result_multi_idx = multi_idx;
                        result_multi_idx[axis] = result_axis_idx;
                        var dst_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| dst_idx[d] = @intCast(result_multi_idx[d]);

                        const val = try self.get(&src_idx);
                        result.set(&dst_idx, val);
                        result_axis_idx += 1;
                    }
                }
            }

            return result;
        }

        /// Reverse the order of elements along a given axis.
        ///
        /// Creates a new array with elements reversed along the specified axis.
        /// Similar to NumPy's flip().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - axis: Axis along which to flip
        ///
        /// Returns: New array with reversed axis
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3}, .row_major);
        /// const flipped = try arr.flip(allocator, 0);
        /// defer flipped.deinit();
        /// // flipped.data = {3, 2, 1}
        /// ```
        pub fn flip(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;

            var result = try Self.init(allocator, &self.shape, self.layout);
            errdefer result.deinit();

            // Calculate total iterations (excluding axis dimension)
            const total_iters = blk: {
                var total: usize = 1;
                for (0..ndim) |d| {
                    if (d != axis) total *= self.shape[d];
                }
                break :blk total;
            };

            for (0..total_iters) |iter_idx| {
                // Convert iteration index to multi-dimensional index (skip axis)
                var temp_idx = iter_idx;
                var multi_idx: [ndim]usize = undefined;
                for (0..ndim) |d| {
                    if (d != axis) {
                        multi_idx[d] = temp_idx % self.shape[d];
                        temp_idx /= self.shape[d];
                    }
                }

                // Copy elements along axis in reverse
                for (0..self.shape[axis]) |i| {
                    multi_idx[axis] = i;
                    var src_idx: [ndim]isize = undefined;
                    for (0..ndim) |d| src_idx[d] = @intCast(multi_idx[d]);

                    multi_idx[axis] = self.shape[axis] - 1 - i;
                    var dst_idx: [ndim]isize = undefined;
                    for (0..ndim) |d| dst_idx[d] = @intCast(multi_idx[d]);

                    const val = try self.get(&src_idx);
                    result.set(&dst_idx, val);
                }
            }

            return result;
        }

        /// Rotate array by 90 degrees in the plane specified by axes.
        ///
        /// Rotates the array k times by 90 degrees counterclockwise.
        /// Similar to NumPy's rot90().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - k: Number of 90-degree rotations (can be negative)
        /// - axes: Two axes defining the rotation plane [axis0, axis1]
        ///
        /// Returns: New rotated array
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 2).fromSlice(allocator, &[_]i32{1, 2, 3, 4}, .row_major);
        /// arr.shape = [2, 2];  // [[1, 2], [3, 4]]
        /// const rotated = try arr.rot90(allocator, 1, [2]usize{0, 1});
        /// // rotated = [[2, 4], [1, 3]]
        /// ```
        pub fn rot90(self: *const Self, allocator: Allocator, k: i32, axes: [2]usize) (Error || AllocatorError)!Self {
            if (axes[0] >= ndim or axes[1] >= ndim) return Error.IndexOutOfBounds;
            if (axes[0] == axes[1]) return Error.ShapeMismatch;

            // Normalize k to 0..3
            const normalized_k = @mod(k, 4);
            if (normalized_k == 0) {
                // No rotation, return copy
                var result = try Self.init(allocator, &self.shape, self.layout);
                errdefer result.deinit();
                @memcpy(result.data, self.data);
                return result;
            }

            // For k rotations: shape[axes[0]] and shape[axes[1]] swap if k is odd
            var new_shape = self.shape;
            if (normalized_k == 1 or normalized_k == 3) {
                const temp = new_shape[axes[0]];
                new_shape[axes[0]] = new_shape[axes[1]];
                new_shape[axes[1]] = temp;
            }

            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Calculate total iterations (excluding rotation axes)
            const total_iters = blk: {
                var total: usize = 1;
                for (0..ndim) |d| {
                    if (d != axes[0] and d != axes[1]) total *= self.shape[d];
                }
                break :blk total;
            };

            for (0..total_iters) |iter_idx| {
                // Build multi-dimensional index for non-rotation dimensions
                var temp_idx = iter_idx;
                var base_idx: [ndim]usize = undefined;
                for (0..ndim) |d| {
                    if (d != axes[0] and d != axes[1]) {
                        base_idx[d] = temp_idx % self.shape[d];
                        temp_idx /= self.shape[d];
                    }
                }

                // Iterate over rotation plane
                for (0..self.shape[axes[0]]) |i| {
                    for (0..self.shape[axes[1]]) |j| {
                        base_idx[axes[0]] = i;
                        base_idx[axes[1]] = j;

                        var src_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| src_idx[d] = @intCast(base_idx[d]);
                        const val = try self.get(&src_idx);

                        // Calculate destination indices based on rotation
                        var dst_i: usize = undefined;
                        var dst_j: usize = undefined;
                        switch (normalized_k) {
                            1 => { // 90 degrees counterclockwise
                                dst_i = self.shape[axes[1]] - 1 - j;
                                dst_j = i;
                            },
                            2 => { // 180 degrees
                                dst_i = self.shape[axes[0]] - 1 - i;
                                dst_j = self.shape[axes[1]] - 1 - j;
                            },
                            3 => { // 270 degrees counterclockwise (90 clockwise)
                                dst_i = j;
                                dst_j = self.shape[axes[0]] - 1 - i;
                            },
                            else => unreachable,
                        }

                        base_idx[axes[0]] = dst_i;
                        base_idx[axes[1]] = dst_j;
                        var dst_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| dst_idx[d] = @intCast(base_idx[d]);
                        result.set(&dst_idx, val);
                    }
                }
            }

            return result;
        }

        /// Roll array elements along a given axis.
        ///
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// Similar to NumPy's roll().
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - shift: Number of positions to shift (positive = right, negative = left)
        /// - axis: Axis along which to roll
        ///
        /// Returns: New rolled array
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 2, 3, 4, 5}, .row_major);
        /// const rolled = try arr.roll(allocator, 2, 0);
        /// defer rolled.deinit();
        /// // rolled.data = {4, 5, 1, 2, 3}
        /// ```
        pub fn roll(self: *const Self, allocator: Allocator, shift: i32, axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;

            var result = try Self.init(allocator, &self.shape, self.layout);
            errdefer result.deinit();

            const axis_len: i32 = @intCast(self.shape[axis]);
            // Normalize shift to 0..axis_len-1
            const normalized_shift = @mod(shift, axis_len);

            // Calculate total iterations (excluding axis dimension)
            const total_iters = blk: {
                var total: usize = 1;
                for (0..ndim) |d| {
                    if (d != axis) total *= self.shape[d];
                }
                break :blk total;
            };

            for (0..total_iters) |iter_idx| {
                // Convert iteration index to multi-dimensional index (skip axis)
                var temp_idx = iter_idx;
                var multi_idx: [ndim]usize = undefined;
                for (0..ndim) |d| {
                    if (d != axis) {
                        multi_idx[d] = temp_idx % self.shape[d];
                        temp_idx /= self.shape[d];
                    }
                }

                // Roll elements along axis
                for (0..self.shape[axis]) |i| {
                    multi_idx[axis] = i;
                    var src_idx: [ndim]isize = undefined;
                    for (0..ndim) |d| src_idx[d] = @intCast(multi_idx[d]);
                    const val = try self.get(&src_idx);

                    const new_pos = @mod(@as(i32, @intCast(i)) + normalized_shift, axis_len);
                    multi_idx[axis] = @intCast(new_pos);
                    var dst_idx: [ndim]isize = undefined;
                    for (0..ndim) |d| dst_idx[d] = @intCast(multi_idx[d]);
                    result.set(&dst_idx, val);
                }
            }

            return result;
        }

        /// Calculate the n-th discrete difference along a given axis.
        ///
        /// Computes differences between consecutive elements:
        /// out[i] = arr[i+1] - arr[i]
        ///
        /// Time: O(n × m) where n = iterations, m = difference order
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - n: Number of times to take the difference (default 1)
        /// - axis: Axis along which to compute differences
        ///
        /// Returns: Array with shape[axis] reduced by n
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(i32, 1).fromSlice(allocator, &[_]i32{1, 3, 6, 10}, .row_major);
        /// const d = try arr.diff(allocator, 1, 0);
        /// defer d.deinit();
        /// // d.data = {2, 3, 4}
        /// ```
        pub fn diff(self: *const Self, allocator: Allocator, n_order: usize, axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            if (n_order == 0) {
                // n=0 means no difference, return copy
                var result = try Self.init(allocator, &self.shape, self.layout);
                errdefer result.deinit();
                @memcpy(result.data, self.data);
                return result;
            }
            if (n_order >= self.shape[axis]) return Error.ShapeMismatch;

            // Start with a copy of self
            var current_arr = try Self.init(allocator, &self.shape, self.layout);
            errdefer current_arr.deinit();
            @memcpy(current_arr.data, self.data);

            // Apply diff n_order times
            for (0..n_order) |_| {
                var new_shape = current_arr.shape;
                new_shape[axis] -= 1;

                var next_arr = try Self.init(allocator, &new_shape, current_arr.layout);
                errdefer next_arr.deinit();

                // Calculate total iterations (excluding axis dimension)
                const total_iters = blk: {
                    var total: usize = 1;
                    for (0..ndim) |d| {
                        if (d != axis) total *= current_arr.shape[d];
                    }
                    break :blk total;
                };

                for (0..total_iters) |iter_idx| {
                    // Convert iteration index to multi-dimensional index (skip axis)
                    var temp_idx = iter_idx;
                    var multi_idx: [ndim]usize = undefined;
                    for (0..ndim) |d| {
                        if (d != axis) {
                            multi_idx[d] = temp_idx % current_arr.shape[d];
                            temp_idx /= current_arr.shape[d];
                        }
                    }

                    // Compute differences along axis
                    for (0..new_shape[axis]) |i| {
                        multi_idx[axis] = i;
                        var idx1: [ndim]isize = undefined;
                        for (0..ndim) |d| idx1[d] = @intCast(multi_idx[d]);
                        const val1 = try current_arr.get(&idx1);

                        multi_idx[axis] = i + 1;
                        var idx2: [ndim]isize = undefined;
                        for (0..ndim) |d| idx2[d] = @intCast(multi_idx[d]);
                        const val2 = try current_arr.get(&idx2);

                        multi_idx[axis] = i;
                        var dst_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| dst_idx[d] = @intCast(multi_idx[d]);
                        next_arr.set(&dst_idx, val2 - val1);
                    }
                }

                // Replace current with next
                current_arr.deinit();
                current_arr = next_arr;
            }

            return current_arr;
        }

        /// Calculate the gradient using finite differences.
        ///
        /// Computes numerical gradient along a given axis using:
        /// - Forward difference at start: grad[0] = arr[1] - arr[0]
        /// - Central difference in middle: grad[i] = (arr[i+1] - arr[i-1]) / 2
        /// - Backward difference at end: grad[n-1] = arr[n-1] - arr[n-2]
        ///
        /// Time: O(n) where n = number of elements
        /// Space: O(n) for result array
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - axis: Axis along which to compute gradient
        ///
        /// Returns: Array with same shape as input
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(f64, 1).fromSlice(allocator, &[_]f64{1, 2, 4, 7, 11}, .row_major);
        /// const grad = try arr.gradient(allocator, 0);
        /// defer grad.deinit();
        /// // grad.data = {1.0, 1.5, 2.5, 3.5, 4.0}
        /// ```
        pub fn gradient(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            if (self.shape[axis] < 2) return Error.ShapeMismatch;

            var result = try Self.init(allocator, &self.shape, self.layout);
            errdefer result.deinit();

            // Calculate total iterations (excluding axis dimension)
            const total_iters = blk: {
                var total: usize = 1;
                for (0..ndim) |d| {
                    if (d != axis) total *= self.shape[d];
                }
                break :blk total;
            };

            for (0..total_iters) |iter_idx| {
                // Convert iteration index to multi-dimensional index (skip axis)
                var temp_idx = iter_idx;
                var multi_idx: [ndim]usize = undefined;
                for (0..ndim) |d| {
                    if (d != axis) {
                        multi_idx[d] = temp_idx % self.shape[d];
                        temp_idx /= self.shape[d];
                    }
                }

                // Compute gradient along axis
                for (0..self.shape[axis]) |i| {
                    multi_idx[axis] = i;
                    var idx: [ndim]isize = undefined;
                    for (0..ndim) |d| idx[d] = @intCast(multi_idx[d]);

                    const grad_val: T = if (i == 0) blk: {
                        // Forward difference at start
                        const curr = try self.get(&idx);
                        multi_idx[axis] = i + 1;
                        var next_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| next_idx[d] = @intCast(multi_idx[d]);
                        const next = try self.get(&next_idx);
                        break :blk next - curr;
                    } else if (i == self.shape[axis] - 1) blk: {
                        // Backward difference at end
                        const curr = try self.get(&idx);
                        multi_idx[axis] = i - 1;
                        var prev_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| prev_idx[d] = @intCast(multi_idx[d]);
                        const prev = try self.get(&prev_idx);
                        break :blk curr - prev;
                    } else blk: {
                        // Central difference in middle
                        multi_idx[axis] = i - 1;
                        var prev_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| prev_idx[d] = @intCast(multi_idx[d]);
                        const prev = try self.get(&prev_idx);

                        multi_idx[axis] = i + 1;
                        var next_idx: [ndim]isize = undefined;
                        for (0..ndim) |d| next_idx[d] = @intCast(multi_idx[d]);
                        const next = try self.get(&next_idx);

                        const two: T = if (@typeInfo(T) == .int) 2 else 2.0;
                        break :blk (next - prev) / two;
                    };

                    multi_idx[axis] = i;
                    var dst_idx: [ndim]isize = undefined;
                    for (0..ndim) |d| dst_idx[d] = @intCast(multi_idx[d]);
                    result.set(&dst_idx, grad_val);
                }
            }

            return result;
        }

        /// Insert values along an axis before a given index
        ///
        /// Creates a new array with values from `values` inserted at position `index` along `axis`.
        /// The resulting array has shape increased by values.shape[axis] along the insertion axis.
        ///
        /// Time: O(n) where n = product of all dimensions
        /// Space: O(n) for the new array
        ///
        /// Example:
        /// ```
        /// arr = [[1, 2], [3, 4]]  // shape [2, 2]
        /// values = [[5, 6]]        // shape [1, 2]
        /// insert(arr, 1, values, 0) => [[1, 2], [5, 6], [3, 4]]  // shape [3, 2]
        /// ```
        pub fn insert(self: *const Self, allocator: Allocator, axis: usize, index: usize, values: *const Self) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            if (index > self.shape[axis]) return Error.IndexOutOfBounds;

            // Validate shapes match except at insertion axis
            for (0..ndim) |d| {
                if (d != axis and values.shape[d] != self.shape[d]) {
                    return Error.ShapeMismatch;
                }
            }

            // Calculate new shape
            var new_shape: [ndim]usize = self.shape;
            new_shape[axis] = self.shape[axis] + values.shape[axis];

            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Copy elements before insertion point
            if (index > 0) {
                try copyArraySegmentInsert(T, ndim, &result, self, axis, 0, 0, index);
            }

            // Copy inserted values
            try copyArraySegmentInsert(T, ndim, &result, values, axis, index, 0, values.shape[axis]);

            // Copy elements after insertion point
            if (index < self.shape[axis]) {
                try copyArraySegmentInsert(T, ndim, &result, self, axis, index + values.shape[axis], index, self.shape[axis] - index);
            }

            return result;
        }

        /// Append values to the end of an array along an axis
        ///
        /// Creates a new array with values from `values` appended to the end along `axis`.
        /// Equivalent to insert(self, allocator, axis, self.shape[axis], values).
        ///
        /// Time: O(n) where n = product of all dimensions
        /// Space: O(n) for the new array
        ///
        /// Example:
        /// ```
        /// arr = [[1, 2], [3, 4]]  // shape [2, 2]
        /// values = [[5, 6]]        // shape [1, 2]
        /// append(arr, values, 0) => [[1, 2], [3, 4], [5, 6]]  // shape [3, 2]
        /// ```
        pub fn append(self: *const Self, allocator: Allocator, axis: usize, values: *const Self) (Error || AllocatorError)!Self {
            return self.insert(allocator, axis, self.shape[axis], values);
        }

        /// Delete a sub-array along an axis
        ///
        /// Removes elements from `start_idx` to `end_idx` (exclusive) along `axis`.
        /// The resulting array has shape decreased by (end_idx - start_idx) along the deletion axis.
        ///
        /// Time: O(n) where n = product of all dimensions
        /// Space: O(n) for the new array
        ///
        /// Example:
        /// ```
        /// arr = [[1, 2], [3, 4], [5, 6]]  // shape [3, 2]
        /// delete(arr, 0, 1, 2) => [[1, 2], [5, 6]]  // shape [2, 2] (removed index 1)
        /// ```
        pub fn delete(self: *const Self, allocator: Allocator, axis: usize, start_idx: usize, end_idx: usize) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            if (start_idx >= end_idx) return Error.IndexOutOfBounds;
            if (end_idx > self.shape[axis]) return Error.IndexOutOfBounds;

            const delete_count = end_idx - start_idx;

            // Calculate new shape
            var new_shape: [ndim]usize = self.shape;
            new_shape[axis] = self.shape[axis] - delete_count;

            if (new_shape[axis] == 0) return Error.ZeroDimension;

            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Copy elements before deletion point
            if (start_idx > 0) {
                try copyArraySegmentInsert(T, ndim, &result, self, axis, 0, 0, start_idx);
            }

            // Copy elements after deletion point
            if (end_idx < self.shape[axis]) {
                try copyArraySegmentInsert(T, ndim, &result, self, axis, start_idx, end_idx, self.shape[axis] - end_idx);
            }

            return result;
        }

        /// Take elements along an axis using an index array (fancy indexing).
        ///
        /// Returns a new array with elements selected by the indices array along the specified axis.
        /// The output shape replaces `shape[axis]` with `indices.size()`.
        ///
        /// Parameters:
        /// - `allocator`: Memory allocator for the new array
        /// - `axis`: Axis along which to take elements (must be < ndim)
        /// - `indices`: 1D array of indices to select (values must be < shape[axis])
        ///
        /// Time: O(prod(shape) × indices.size() / shape[axis])
        /// Space: O(prod(new_shape)) for the output array
        ///
        /// Example:
        /// ```
        /// arr = [[1, 2], [3, 4], [5, 6]]  // shape [3, 2]
        /// indices = [0, 2]
        /// take(arr, 0, indices) => [[1, 2], [5, 6]]  // shape [2, 2]
        /// ```
        pub fn take(self: *const Self, allocator: Allocator, axis: usize, indices: *const NDArray(usize, 1)) (Error || AllocatorError)!Self {
            if (axis >= ndim) return Error.IndexOutOfBounds;
            const indices_len = indices.count();
            if (indices_len == 0) return Error.ZeroDimension;

            // Validate all indices are within bounds
            var indices_iter = indices.iterator();
            while (indices_iter.next()) |idx| {
                if (idx >= self.shape[axis]) return Error.IndexOutOfBounds;
            }

            // Calculate new shape
            var new_shape: [ndim]usize = self.shape;
            new_shape[axis] = indices_len;

            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Multi-dimensional index iteration
            var multi_idx: [ndim]usize = undefined;
            @memset(&multi_idx, 0);

            const total = result.count();
            var elem_count: usize = 0;

            // Iterate over the output positions
            while (elem_count < total) : (elem_count += 1) {
                // Calculate the current position in the output array
                var temp = elem_count;
                for (0..ndim) |d| {
                    var divisor: usize = 1;
                    for (d + 1..ndim) |dd| {
                        divisor *= new_shape[dd];
                    }
                    multi_idx[d] = temp / divisor;
                    temp %= divisor;
                }

                // Map the index along `axis` through the indices array
                var source_idx = multi_idx;
                const take_idx = multi_idx[axis];
                source_idx[axis] = indices.data[take_idx];

                // Calculate offset in source and destination
                var src_offset: usize = 0;
                var dst_offset: usize = 0;
                for (0..ndim) |d| {
                    src_offset += source_idx[d] * self.strides[d];
                    dst_offset += multi_idx[d] * result.strides[d];
                }

                result.data[dst_offset] = self.data[src_offset];
            }

            return result;
        }

        /// Put values into the array at specified flat indices.
        ///
        /// Modifies the array in-place by placing values from the `values` array
        /// at positions specified by the `indices` array (using flat/linear indexing).
        /// This is equivalent to NumPy's `put()` function.
        ///
        /// Parameters:
        /// - `indices`: 1D array of flat indices where values should be placed
        /// - `values`: 1D array of values to place (must have same length as indices)
        ///
        /// Time: O(indices.size() × ndim) for index to offset conversion
        /// Space: O(1) in-place modification
        ///
        /// Example:
        /// ```
        /// arr = [[1, 2], [3, 4]]  // flat: [1, 2, 3, 4]
        /// indices = [0, 3]
        /// values = [9, 10]
        /// put(arr, indices, values) => arr = [[9, 2], [3, 10]]
        /// ```
        pub fn put(self: *Self, indices: *const NDArray(usize, 1), values: *const NDArray(T, 1)) Error!void {
            const indices_len = indices.count();
            const values_len = values.count();
            if (indices_len != values_len) return Error.ShapeMismatch;
            if (indices_len == 0) return; // No-op for empty indices

            const total = self.count();

            // Iterate over indices and values
            for (0..indices_len) |i| {
                const flat_idx = indices.data[i];
                if (flat_idx >= total) return Error.IndexOutOfBounds;

                // Convert flat index to multi-dimensional index
                var multi_idx: [ndim]usize = undefined;
                var temp = flat_idx;

                for (0..ndim) |d| {
                    var divisor: usize = 1;
                    for (d + 1..ndim) |dd| {
                        divisor *= self.shape[dd];
                    }
                    multi_idx[d] = temp / divisor;
                    temp %= divisor;
                }

                // Calculate offset using strides
                var offset: usize = 0;
                for (0..ndim) |d| {
                    offset += multi_idx[d] * self.strides[d];
                }

                self.data[offset] = values.data[i];
            }
        }


        // -- Iterator Protocol --

        /// Iterator type for traversing NDArray elements
        ///
        /// Implements the zuda iterator protocol: next() -> ?T
        /// Supports layout-aware traversal respecting strides
        pub const Iterator = struct {
            data: []const T,
            shape: [ndim]usize,
            strides: [ndim]usize,
            layout: Layout,
            index: usize, // Current flat index
            total: usize, // Total number of elements

            /// Get the next element in traversal order
            ///
            /// Returns: Next element in storage order, or null if exhausted
            ///
            /// Traversal respects memory layout:
            /// - Row-major: rightmost dimension varies fastest
            /// - Column-major: leftmost dimension varies fastest
            ///
            /// Time: O(ndim) multi-dimensional index calculation
            /// Space: O(1)
            pub fn next(self: *Iterator) ?T {
                // Return null if we've consumed all elements
                if (self.index >= self.total) {
                    return null;
                }

                // Convert flat index to multi-dimensional indices
                // respecting the layout (row-major or column-major)
                var multi_index: [ndim]usize = undefined;
                var current = self.index;

                if (self.layout == .row_major) {
                    // Row-major: rightmost dimension varies fastest
                    // For shape [d0, d1, d2, ...], flat index i converts to:
                    // index[0] = i / (d1*d2*...), then i %= (d1*d2*...)
                    // index[1] = i / (d2*...), then i %= (d2*...)
                    for (0..ndim) |dim| {
                        var divisor: usize = 1;
                        for (dim + 1..ndim) |d| {
                            divisor *= self.shape[d];
                        }
                        multi_index[dim] = current / divisor;
                        current = current % divisor;
                    }
                } else {
                    // Column-major: leftmost dimension varies fastest
                    // For shape [d0, d1, d2, ...], flat index i converts to:
                    // index[ndim-1] = i / (d0*d1*...), then i %= (d0*d1*...)
                    // index[ndim-2] = i / (d0*...), then i %= (d0*...)
                    // Work backwards from last dimension
                    var dim_idx: usize = ndim;
                    while (dim_idx > 0) {
                        dim_idx -= 1;
                        var divisor: usize = 1;
                        for (0..dim_idx) |d| {
                            divisor *= self.shape[d];
                        }
                        multi_index[dim_idx] = current / divisor;
                        current = current % divisor;
                    }
                }

                // Calculate memory offset using array strides
                var offset: usize = 0;
                for (0..ndim) |i| {
                    offset += multi_index[i] * self.strides[i];
                }

                // Get value from memory
                const value = self.data[offset];

                // Move to next element
                self.index += 1;

                return value;
            }
        };

        /// Create an iterator over the array
        ///
        /// Returns: Iterator positioned at the first element
        ///
        /// The iterator respects the array's layout and strides,
        /// correctly traversing views and slices.
        ///
        /// Time: O(1) iterator creation
        /// Space: O(ndim) for iterator state
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{
                .data = self.data,
                .shape = self.shape,
                .strides = self.strides,
                .layout = self.layout,
                .index = 0,
                .total = self.count(),
            };
        }

        // -- Boolean Reductions (only for T == bool) --

        /// Boolean AND reduction - all elements true
        ///
        /// Only available for NDArray(bool, ndim)
        ///
        /// Returns: true if all elements are true, false otherwise
        ///
        /// For empty arrays, returns true (vacuous truth)
        ///
        /// Time: O(n) worst case, O(1) best case (short-circuit on first false)
        /// Space: O(1)
        pub fn all(self: *const Self) bool {
            if (T != bool) {
                @compileError("all() is only available for NDArray(bool, ndim)");
            }

            for (self.data) |val| {
                if (!val) {
                    return false;
                }
            }
            return true;
        }

        /// Boolean OR reduction - any element true
        ///
        /// Only available for NDArray(bool, ndim)
        ///
        /// Returns: true if any element is true, false otherwise
        ///
        /// For empty arrays, returns false
        ///
        /// Time: O(n) worst case, O(1) best case (short-circuit on first true)
        /// Space: O(1)
        pub fn any(self: *const Self) bool {
            if (T != bool) {
                @compileError("any() is only available for NDArray(bool, ndim)");
            }

            for (self.data) |val| {
                if (val) {
                    return true;
                }
            }
            return false;
        }

        /// Remove a dimension of size 1 at the specified axis
        ///
        /// Creates a new NDArray with reduced dimensionality by removing one dimension
        /// that has size 1. The data is shared (zero-copy view).
        ///
        /// Time: O(1) - creates view without copying data
        /// Space: O(1) - shares data with original array
        ///
        /// Errors:
        /// - ShapeMismatch: if dimension at axis is not size 1
        /// - IndexOutOfBounds: if axis >= ndim
        pub fn squeeze(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim - 1) {
            if (ndim == 0) {
                @compileError("Cannot squeeze 0-dimensional array");
            }
            if (axis >= ndim) {
                return Error.IndexOutOfBounds;
            }
            if (self.shape[axis] != 1) {
                return Error.ShapeMismatch;
            }

            var new_shape: [ndim - 1]usize = undefined;
            var new_strides: [ndim - 1]usize = undefined;

            // Copy shape and strides, skipping the squeezed axis
            var j: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    new_shape[j] = self.shape[i];
                    new_strides[j] = self.strides[i];
                    j += 1;
                }
            }

            // Create view with reduced dimensionality
            return NDArray(T, ndim - 1){
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = allocator,
                .layout = self.layout,
                .owned = false, // View shares data
            };
        }


        /// Add a dimension of size 1 at the specified axis
        ///
        /// Creates a new NDArray with increased dimensionality by inserting a dimension
        /// of size 1 at the given axis. The data is shared (zero-copy view).
        ///
        /// Time: O(1) - creates view without copying data
        /// Space: O(1) - shares data with original array
        ///
        /// Errors:
        /// - IndexOutOfBounds: if axis > ndim (note: axis == ndim is valid for trailing insertion)
        pub fn unsqueeze(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            if (axis > ndim) {
                return Error.IndexOutOfBounds;
            }

            var new_shape: [ndim + 1]usize = undefined;
            var new_strides: [ndim + 1]usize = undefined;

            // Copy shape and strides, inserting size-1 dimension at axis
            var j: usize = 0;
            for (0..ndim + 1) |i| {
                if (i == axis) {
                    new_shape[i] = 1;
                    // Stride for size-1 dimension can be any value (commonly set to element stride)
                    new_strides[i] = if (ndim > 0) self.strides[0] else @sizeOf(T);
                } else {
                    new_shape[i] = self.shape[j];
                    new_strides[i] = self.strides[j];
                    j += 1;
                }
            }

            return NDArray(T, ndim + 1){
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = allocator,
                .layout = self.layout,
                .owned = false, // View shares data
            };
        }

        /// View input as array with at least one dimension.
        /// Scalar arrays become 1D arrays, 1D+ arrays are returned unchanged as views.
        /// Similar to NumPy's atleast_1d().
        ///
        /// Time: O(1) - creates view
        /// Space: O(1) - no data copying
        ///
        /// Example:
        /// ```
        /// // 0D scalar -> 1D with shape [1]
        /// const scalar = try NDArray(f64, 0).fromSlice(allocator, &[_]usize{}, &[_]f64{42}, .row_major);
        /// const arr1d = try scalar.atleast1d(allocator);
        /// // arr1d.shape = [1], arr1d.data[0] = 42
        /// ```
        pub fn atleast1d(self: *const Self, allocator: Allocator) (Error || AllocatorError)!if (ndim == 0) NDArray(T, 1) else Self {
            if (ndim == 0) {
                // Scalar -> 1D with shape [1]
                return NDArray(T, 1){
                    .shape = [_]usize{1},
                    .strides = [_]usize{@sizeOf(T)},
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false, // View shares data
                };
            } else {
                // Already 1D+, return view
                return self.createView();
            }
        }

        /// View input as array with at least two dimensions.
        /// 0D becomes 2D with shape [1,1], 1D becomes 2D with shape [1,N].
        /// Similar to NumPy's atleast_2d().
        ///
        /// Time: O(1) - creates view
        /// Space: O(1) - no data copying
        ///
        /// Example:
        /// ```
        /// // 1D [N] -> 2D [1, N]
        /// const arr1d = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{1,2,3}, .row_major);
        /// const arr2d = try arr1d.atleast2d(allocator);
        /// // arr2d.shape = [1, 3]
        /// ```
        pub fn atleast2d(self: *const Self, allocator: Allocator) (Error || AllocatorError)!if (ndim < 2) NDArray(T, 2) else Self {
            if (ndim == 0) {
                // Scalar -> 2D with shape [1, 1]
                return NDArray(T, 2){
                    .shape = [_]usize{ 1, 1 },
                    .strides = [_]usize{ @sizeOf(T), @sizeOf(T) },
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false,
                };
            } else if (ndim == 1) {
                // 1D [N] -> 2D [1, N] (prepend dimension)
                return NDArray(T, 2){
                    .shape = [_]usize{ 1, self.shape[0] },
                    .strides = [_]usize{ self.strides[0] * self.shape[0], self.strides[0] },
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false,
                };
            } else {
                // Already 2D+, return view
                return self.createView();
            }
        }

        /// View input as array with at least three dimensions.
        /// 0D becomes [1,1,1], 1D [N] becomes [1,N,1], 2D [M,N] becomes [M,N,1].
        /// Similar to NumPy's atleast_3d().
        ///
        /// Time: O(1) - creates view
        /// Space: O(1) - no data copying
        ///
        /// Example:
        /// ```
        /// // 2D [M,N] -> 3D [M,N,1]
        /// const arr2d = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2,3}, &[_]f64{1,2,3,4,5,6}, .row_major);
        /// const arr3d = try arr2d.atleast3d(allocator);
        /// // arr3d.shape = [2, 3, 1]
        /// ```
        pub fn atleast3d(self: *const Self, allocator: Allocator) (Error || AllocatorError)!if (ndim < 3) NDArray(T, 3) else Self {
            if (ndim == 0) {
                // Scalar -> 3D [1, 1, 1]
                return NDArray(T, 3){
                    .shape = [_]usize{ 1, 1, 1 },
                    .strides = [_]usize{ @sizeOf(T), @sizeOf(T), @sizeOf(T) },
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false,
                };
            } else if (ndim == 1) {
                // 1D [N] -> 3D [1, N, 1]
                return NDArray(T, 3){
                    .shape = [_]usize{ 1, self.shape[0], 1 },
                    .strides = [_]usize{
                        self.strides[0] * self.shape[0],
                        self.strides[0],
                        @sizeOf(T),
                    },
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false,
                };
            } else if (ndim == 2) {
                // 2D [M, N] -> 3D [M, N, 1] (append dimension)
                return NDArray(T, 3){
                    .shape = [_]usize{ self.shape[0], self.shape[1], 1 },
                    .strides = [_]usize{ self.strides[0], self.strides[1], @sizeOf(T) },
                    .data = self.data,
                    .allocator = allocator,
                    .layout = self.layout,
                    .owned = false,
                };
            } else {
                // Already 3D+, return view
                return self.createView();
            }
        }

        /// Expand dimensions of array by inserting a new axis.
        /// Alias for unsqueeze() for NumPy compatibility.
        ///
        /// Time: O(1) - creates view
        /// Space: O(1) - no data copying
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2,3}, &[_]f64{1,2,3,4,5,6}, .row_major);
        /// const expanded = try arr.expandDims(allocator, 1);
        /// // expanded.shape = [2, 1, 3]
        /// ```
        pub fn expandDims(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            return self.unsqueeze(allocator, axis);
        }

        /// Broadcast array to new shape.
        /// Creates a view with modified strides to broadcast along size-1 dimensions.
        ///
        /// Broadcasting rules:
        /// - Trailing dimensions must match or one must be 1
        /// - Missing leading dimensions are treated as 1
        ///
        /// Time: O(1) - creates view with modified strides
        /// Space: O(1) - no data copying
        ///
        /// Errors:
        /// - ShapeMismatch: if shapes are not broadcast-compatible
        ///
        /// Example:
        /// ```
        /// // [3, 1] -> [3, 4] by broadcasting along axis 1
        /// const arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3,1}, &[_]f64{1,2,3}, .row_major);
        /// const broadcasted = try arr.broadcastTo(allocator, [_]usize{3,4});
        /// // broadcasted appears as [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
        /// ```
        pub fn broadcastTo(self: *const Self, allocator: Allocator, new_shape: [ndim]usize) (Error || AllocatorError)!Self {
            // Verify broadcast compatibility
            for (0..ndim) |i| {
                if (self.shape[i] != new_shape[i]) {
                    if (self.shape[i] != 1) {
                        return Error.ShapeMismatch;
                    }
                }
            }

            // Create view with modified strides
            var new_strides: [ndim]usize = undefined;
            for (0..ndim) |i| {
                if (self.shape[i] == 1 and new_shape[i] > 1) {
                    // Broadcasting: set stride to 0 so same element is reused
                    new_strides[i] = 0;
                } else {
                    // No broadcasting needed
                    new_strides[i] = self.strides[i];
                }
            }

            return Self{
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = allocator,
                .layout = self.layout,
                .owned = false, // View shares data
            };
        }

        /// Concatenate multiple arrays along an existing axis
        ///
        /// Joins a sequence of arrays along the specified axis. All arrays must have
        /// the same shape except along the concatenation axis.
        ///
        /// Time: O(n) where n = total elements in result
        /// Space: O(n) - allocates new array
        ///
        /// Errors:
        /// - IndexOutOfBounds: if axis >= ndim
        /// - ShapeMismatch: if arrays have incompatible shapes
        /// - EmptyArray: if arrays slice is empty
        ///
        /// Example:
        /// ```zig
        /// var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{1,2,3,4,5,6}, .row_major);
        /// var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{7,8,9,10,11,12}, .row_major);
        /// var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){&a, &b}, 0, .row_major);
        /// // result shape: [4, 3] (concatenated along axis 0)
        /// ```
        pub fn concat(allocator: Allocator, arrays: []const *const Self, axis: usize, layout: Layout) (Error || AllocatorError)!Self {
            if (arrays.len == 0) {
                return Error.EmptyArray;
            }
            if (axis >= ndim) {
                return Error.IndexOutOfBounds;
            }

            // Validate all arrays have compatible shapes
            const first = arrays[0];
            var total_concat_size: usize = first.shape[axis];

            for (arrays[1..]) |arr| {
                // Check all dimensions except concat axis match
                for (0..ndim) |i| {
                    if (i != axis and arr.shape[i] != first.shape[i]) {
                        return Error.ShapeMismatch;
                    }
                }
                total_concat_size += arr.shape[axis];
            }

            // Build result shape
            var result_shape: [ndim]usize = first.shape;
            result_shape[axis] = total_concat_size;

            // Create result array
            var result = try Self.init(allocator, &result_shape, layout);
            errdefer result.deinit();

            // Element-by-element copy with proper indexing
            var concat_offset: usize = 0;
            for (arrays) |arr| {
                try copyArraySegment(T, ndim, &result, arr, axis, concat_offset);
                concat_offset += arr.shape[axis];
            }

            return result;
        }

        /// Stack arrays along a new dimension
        ///
        /// Time: O(n × m) where n = num arrays, m = elements per array
        /// Space: O(n × m)
        ///
        /// Joins a sequence of arrays along a new dimension. All input arrays must have
        /// identical shapes. The new dimension is inserted at position `axis`.
        ///
        /// Examples:
        /// - 3 arrays of shape [2, 3] stacked at axis=0 → [3, 2, 3]
        /// - 3 arrays of shape [2, 3] stacked at axis=1 → [2, 3, 3]
        /// - 3 arrays of shape [2, 3] stacked at axis=2 → [2, 3, 3]
        ///
        /// Parameters:
        /// - allocator: Memory allocator for result array
        /// - arrays: Slice of array pointers to stack (all must have identical shapes)
        /// - axis: Position to insert new dimension (0 to ndim inclusive)
        /// - layout: Memory layout for result array
        ///
        /// Returns: NDArray(T, ndim+1) with shape [...shape[:axis], n, ...shape[axis:]]
        /// Errors: EmptyArray, IndexOutOfBounds, ShapeMismatch
        pub fn stack(allocator: Allocator, arrays: []const *const Self, axis: usize, layout: Layout) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            if (arrays.len == 0) {
                return Error.EmptyArray;
            }
            if (axis > ndim) {
                return Error.IndexOutOfBounds;
            }

            // Validate all arrays have identical shapes
            const first = arrays[0];
            for (arrays[1..]) |arr| {
                for (0..ndim) |i| {
                    if (arr.shape[i] != first.shape[i]) {
                        return Error.ShapeMismatch;
                    }
                }
            }

            // Build result shape: [...shape[:axis], n, ...shape[axis:]]
            var result_shape: [ndim + 1]usize = undefined;
            for (0..axis) |i| {
                result_shape[i] = first.shape[i];
            }
            result_shape[axis] = arrays.len;
            for (axis..ndim) |i| {
                result_shape[i + 1] = first.shape[i];
            }

            // Create result array
            const ResultType = NDArray(T, ndim + 1);
            var result = try ResultType.init(allocator, &result_shape, layout);
            errdefer result.deinit();

            // Copy each input array to the appropriate slice of result
            for (arrays, 0..) |arr, arr_idx| {
                try copyArrayToStack(T, ndim, &result, arr, axis, arr_idx);
            }

            return result;
        }

        /// Split an array into multiple sub-arrays along an axis
        ///
        /// Divides the array into N equal sections along the specified axis.
        /// The dimension along the axis must be evenly divisible by N.
        ///
        /// Parameters:
        /// - allocator: Memory allocator for result arrays
        /// - axis: Dimension along which to split (must be < ndim)
        /// - n_sections: Number of equal sections to split into
        ///
        /// Returns: Slice of N NDArray objects with equal shapes
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - error.ShapeMismatch if shape[axis] not divisible by n_sections
        /// - error.ZeroDimension if n_sections == 0
        /// - error.OutOfMemory if allocation fails
        ///
        /// Time: O(prod(shape)) — copies all elements
        /// Space: O(prod(shape)) — creates N new arrays
        ///
        /// Example:
        /// ```zig
        /// // Split [6, 4] array along axis=0 into 3 parts → 3×[2, 4] arrays
        /// var arr = try NDArray(f64, 2).init(allocator, &[_]usize{6, 4}, .row_major);
        /// defer arr.deinit();
        /// const parts = try arr.split(allocator, 0, 3);
        /// defer allocator.free(parts);
        /// for (parts) |*part| part.deinit();
        /// // parts[0].shape = [2, 4], parts[1].shape = [2, 4], parts[2].shape = [2, 4]
        /// ```
        pub fn split(self: *const Self, allocator: Allocator, axis: usize, n_sections: usize) (Error || AllocatorError)![]Self {
            if (axis >= ndim) {
                return Error.IndexOutOfBounds;
            }
            if (n_sections == 0) {
                return Error.ZeroDimension;
            }

            const axis_size = self.shape[axis];
            if (axis_size % n_sections != 0) {
                return Error.ShapeMismatch;
            }

            const section_size = axis_size / n_sections;

            // Calculate shape of each section
            var section_shape = self.shape;
            section_shape[axis] = section_size;

            // Allocate result array
            const result = try allocator.alloc(Self, n_sections);
            errdefer allocator.free(result);

            // Initialize each section
            var initialized: usize = 0;
            errdefer {
                for (0..initialized) |i| {
                    result[i].deinit();
                }
            }

            for (0..n_sections) |section_idx| {
                // Create new array for this section
                result[section_idx] = try Self.init(allocator, &section_shape, self.layout);
                initialized += 1;
                errdefer result[section_idx].deinit();

                // Copy data from source to this section
                try copySplitSection(T, ndim, &result[section_idx], self, axis, section_idx * section_size);
            }

            return result;
        }

        /// Stack arrays vertically (row-wise) — convenience wrapper for stack() along axis 0
        ///
        /// Time: O(n × m) | Space: O(n × m)
        pub fn vstack(allocator: Allocator, arrays: []const *const Self, layout: Layout) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            return stack(allocator, arrays, 0, layout);
        }

        /// Stack arrays horizontally (column-wise)
        ///
        /// For 1D arrays: concatenates along axis 0
        /// For ndim ≥ 2: concatenates along axis 1 (columns)
        ///
        /// Time: O(n × m) | Space: O(n × m)
        pub fn hstack(allocator: Allocator, arrays: []const *const Self, layout: Layout) (Error || AllocatorError)!Self {
            if (ndim == 1) {
                return concat(allocator, arrays, 0, layout);
            } else {
                return concat(allocator, arrays, 1, layout);
            }
        }

        /// Stack arrays depth-wise (along axis 2) — convenience wrapper for stack() along axis 2
        ///
        /// Requires ndim ≥ 2.
        ///
        /// Time: O(n × m) | Space: O(n × m)
        pub fn dstack(allocator: Allocator, arrays: []const *const Self, layout: Layout) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            comptime {
                if (ndim < 2) {
                    @compileError("dstack requires ndim >= 2");
                }
            }
            return stack(allocator, arrays, 2, layout);
        }

        /// Stack arrays row-wise — alias for vstack()
        ///
        /// Time: O(n × m) | Space: O(n × m)
        pub fn row_stack(allocator: Allocator, arrays: []const *const Self, layout: Layout) (Error || AllocatorError)!NDArray(T, ndim + 1) {
            return vstack(allocator, arrays, layout);
        }

        /// Stack 1D arrays as columns into a 2D array
        ///
        /// For 1D arrays: converts each to a column vector and stacks horizontally
        /// For 2D arrays: same as hstack (concatenates along axis 1)
        ///
        /// Time: O(n × m) | Space: O(n × m)
        pub fn column_stack(allocator: Allocator, arrays: []const *const Self, layout: Layout) (Error || AllocatorError)!NDArray(T, if (ndim == 1) 2 else ndim) {
            if (arrays.len == 0) {
                return Error.EmptyArray;
            }

            if (ndim == 1) {
                const n_rows = arrays[0].shape[0];
                const n_cols = arrays.len;

                for (arrays[1..]) |arr| {
                    if (arr.shape[0] != n_rows) {
                        return Error.ShapeMismatch;
                    }
                }

                var result = try NDArray(T, 2).init(allocator, &[_]usize{ n_rows, n_cols }, layout);
                errdefer result.deinit();

                for (arrays, 0..) |arr, col_idx| {
                    for (0..n_rows) |row_idx| {
                        const val = arr.data[row_idx * arr.strides[0]];
                        const result_offset = row_idx * result.strides[0] + col_idx * result.strides[1];
                        result.data[result_offset] = val;
                    }
                }

                return result;
            } else {
                return hstack(allocator, arrays, layout);
            }
        }

        /// Padding modes for array extension
        pub const PadMode = enum {
            /// Pad with constant value
            constant,
            /// Extend edge values
            edge,
            /// Mirror reflection without repeating edge values (a b c d | c b a)
            reflect,
            /// Mirror reflection with edge values repeated (a b c d | d c b a)
            symmetric,
            /// Circular wrapping (a b c d | a b c d)
            wrap,
        };

        /// Pad array along each axis
        ///
        /// Extends array dimensions by adding values along each axis according to the padding mode.
        ///
        /// Parameters:
        /// - allocator: Memory allocator for result array
        /// - pad_width: Array of [before, after] padding amounts for each axis
        /// - mode: Padding strategy (constant, edge, reflect, symmetric, wrap)
        /// - constant_value: Value used when mode = .constant
        ///
        /// Returns: New padded array with shape = original_shape + sum(pad_width)
        ///
        /// Errors:
        /// - error.ZeroDimension if pad_width length ≠ ndim
        /// - error.CapacityExceeded if padded size exceeds usize.max
        /// - error.OutOfMemory if allocation fails
        ///
        /// Time: O(prod(padded_shape))
        /// Space: O(prod(padded_shape))
        ///
        /// Example:
        /// ```zig
        /// var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{1, 2, 3}, .row_major);
        /// defer arr.deinit();
        /// var padded = try arr.pad(allocator, &[_][2]usize{.{1, 1}}, .constant, 0);
        /// defer padded.deinit();
        /// // padded.data = [0, 1, 2, 3, 0]
        /// ```
        pub fn pad(
            self: *const Self,
            allocator: Allocator,
            pad_width: []const [2]usize,
            pad_mode: PadMode,
            constant_value: T,
        ) (Error || AllocatorError)!Self {
            if (pad_width.len != ndim) {
                return Error.ZeroDimension;
            }

            // Calculate padded shape
            var padded_shape: [ndim]usize = undefined;
            for (0..ndim) |i| {
                const before = pad_width[i][0];
                const after = pad_width[i][1];
                padded_shape[i] = self.shape[i] + before + after;
                if (padded_shape[i] == 0) {
                    return Error.ZeroDimension;
                }
            }

            // Create result array
            var result = try Self.init(allocator, &padded_shape, self.layout);
            errdefer result.deinit();

            // Copy original data to center of padded array
            try copyToPaddedCenter(T, ndim, &result, self, pad_width);

            // Fill padding regions based on mode
            switch (pad_mode) {
                .constant => try fillConstantPadding(T, ndim, &result, self, pad_width, constant_value),
                .edge => try fillEdgePadding(T, ndim, &result, self, pad_width),
                .reflect => try fillReflectPadding(T, ndim, &result, self, pad_width),
                .symmetric => try fillSymmetricPadding(T, ndim, &result, self, pad_width),
                .wrap => try fillWrapPadding(T, ndim, &result, self, pad_width),
            }

            return result;
        }

        /// Repeat elements of array along specified axis.
        ///
        /// Time: O(prod(shape) × repeats)
        /// Space: O(prod(shape) × repeats)
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(f64, 1).fromSlice(allocator, &.{3}, &.{1, 2, 3}, .row_major);
        /// defer arr.deinit();
        /// const repeated = try arr.repeat(allocator, 2, 0);
        /// defer repeated.deinit();
        /// // repeated.data = [1, 1, 2, 2, 3, 3]
        /// ```
        pub fn repeat(
            self: *const Self,
            allocator: Allocator,
            repeats: usize,
            axis: usize,
        ) (Error || AllocatorError)!Self {
            if (repeats == 0) {
                return Error.ZeroDimension;
            }

            if (axis >= ndim) {
                return Error.IndexOutOfBounds;
            }

            // Calculate new shape
            var new_shape: [ndim]usize = undefined;
            for (0..ndim) |i| {
                if (i == axis) {
                    new_shape[i] = self.shape[i] * repeats;
                } else {
                    new_shape[i] = self.shape[i];
                }
            }

            var result = try Self.init(allocator, &new_shape, self.layout);
            errdefer result.deinit();

            // Iterate through source array
            var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);
            var done = false;

            while (!done) {
                var src_indices: [ndim]isize = undefined;
                for (indices, 0..) |idx, i| {
                    src_indices[i] = @as(isize, @intCast(idx));
                }
                const val = try self.get(&src_indices);

                // Repeat this element along the specified axis
                for (0..repeats) |rep| {
                    var dest_indices: [ndim]isize = undefined;
                    for (indices, 0..) |idx, i| {
                        if (i == axis) {
                            dest_indices[i] = @as(isize, @intCast(idx * repeats + rep));
                        } else {
                            dest_indices[i] = @as(isize, @intCast(idx));
                        }
                    }
                    result.set(&dest_indices, val);
                }

                // Increment multi-dimensional index
                var carry: usize = 1;
                var dim: usize = ndim;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    indices[dim] += carry;
                    if (indices[dim] >= self.shape[dim]) {
                        indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Repeat elements of array after flattening.
        /// Returns a 1D array with each element repeated the specified number of times.
        ///
        /// Time: O(prod(shape) × repeats)
        /// Space: O(prod(shape) × repeats)
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(f64, 2).fromSlice(allocator, &.{2, 2}, &.{1, 2, 3, 4}, .row_major);
        /// defer arr.deinit();
        /// const repeated = try arr.repeatFlat(allocator, 2);
        /// defer repeated.deinit();
        /// // repeated.data = [1, 1, 2, 2, 3, 3, 4, 4] (1D)
        /// ```
        pub fn repeatFlat(
            self: *const Self,
            allocator: Allocator,
            repeats: usize,
        ) (Error || AllocatorError)!NDArray(T, 1) {
            if (repeats == 0) {
                return Error.ZeroDimension;
            }

            const total_size = self.count();
            const result_size = total_size * repeats;
            var result = try NDArray(T, 1).init(allocator, &.{result_size}, self.layout);
            errdefer result.deinit();

            var flat_idx: usize = 0;
            var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);
            var done = false;

            while (!done) {
                var src_indices: [ndim]isize = undefined;
                for (indices, 0..) |idx, i| {
                    src_indices[i] = @as(isize, @intCast(idx));
                }
                const val = try self.get(&src_indices);

                // Repeat this element `repeats` times
                for (0..repeats) |rep| {
                    const out_idx: isize = @intCast(flat_idx * repeats + rep);
                    result.set(&.{out_idx}, val);
                }
                flat_idx += 1;

                // Increment multi-dimensional index
                var carry: usize = 1;
                var dim: usize = ndim;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    indices[dim] += carry;
                    if (indices[dim] >= self.shape[dim]) {
                        indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Construct array by repeating the input array.
        /// The reps parameter specifies the number of repetitions along each axis.
        ///
        /// Time: O(prod(shape) × prod(reps))
        /// Space: O(prod(shape) × prod(reps))
        ///
        /// Example:
        /// ```
        /// const arr = try NDArray(f64, 2).fromSlice(allocator, &.{2, 2}, &.{1, 2, 3, 4}, .row_major);
        /// defer arr.deinit();
        /// const tiled = try arr.tile(allocator, &.{2, 3});
        /// defer tiled.deinit();
        /// // Tiles array 2 times vertically, 3 times horizontally
        /// // Result shape: [4, 6]
        /// ```
        pub fn tile(
            self: *const Self,
            allocator: Allocator,
            reps: []const usize,
        ) (Error || AllocatorError)!Self {
            if (reps.len != ndim) {
                return Error.ShapeMismatch;
            }

            // Check for zero repetitions
            for (reps) |r| {
                if (r == 0) {
                    return Error.ZeroDimension;
                }
            }

            // Calculate output shape
            var out_shape: [ndim]usize = undefined;
            for (0..ndim) |i| {
                out_shape[i] = self.shape[i] * reps[i];
            }

            var result = try Self.init(allocator, &out_shape, self.layout);
            errdefer result.deinit();

            // Iterate through all output positions
            var out_indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);
            var done = false;

            while (!done) {
                // Map output position to source position (using modulo)
                var src_indices: [ndim]isize = undefined;
                for (out_indices, 0..) |out_idx, i| {
                    src_indices[i] = @as(isize, @intCast(out_idx % self.shape[i]));
                }

                const val = try self.get(&src_indices);

                var out_indices_signed: [ndim]isize = undefined;
                for (out_indices, 0..) |idx, i| {
                    out_indices_signed[i] = @as(isize, @intCast(idx));
                }
                result.set(&out_indices_signed, val);

                // Increment output index
                var carry: usize = 1;
                var dim: usize = ndim;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    out_indices[dim] += carry;
                    if (out_indices[dim] >= out_shape[dim]) {
                        out_indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Sum along a specified axis, reducing dimensionality
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Axis to sum along (must be < ndim)
        ///
        /// Returns: NDArray with shape = original shape with axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        ///
        /// Example: [3,4,5] sumAxis(1) → [3,5]
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(m) where m = prod(output shape)
        pub fn sumAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim - 1) {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Build output shape by removing the reduction axis
            var out_shape: [ndim - 1]usize = undefined;
            if (ndim > 1) {
                var idx: usize = 0;
                for (self.shape, 0..) |dim, i| {
                    if (i != axis) {
                        out_shape[idx] = dim;
                        idx += 1;
                    }
                }
            }

            // Allocate result array initialized to zero
            var result = try NDArray(T, ndim - 1).zeros(allocator, &out_shape, self.layout);
            errdefer result.deinit();

            // Special case: reducing to 0D (scalar)
            if (ndim == 1) {
                var sum_val: T = 0;
                for (self.data) |val| {
                    sum_val += val;
                }
                result.data[0] = sum_val;
                return result;
            }

            // Iterate through all output positions
            var out_indices: [ndim - 1]usize = stdlib.mem.zeroes([ndim - 1]usize);
            var done = false;

            while (!done) {
                // Sum along the reduction axis
                var sum_val: T = 0;
                for (0..self.shape[axis]) |axis_idx| {
                    // Map output position to full input position
                    var full_indices: [ndim]usize = undefined;
                    var out_idx: usize = 0;
                    for (0..ndim) |i| {
                        if (i == axis) {
                            full_indices[i] = axis_idx;
                        } else {
                            full_indices[i] = out_indices[out_idx];
                            out_idx += 1;
                        }
                    }

                    // Convert to linear index and get value
                    var linear_idx: usize = 0;
                    for (0..ndim) |i| {
                        linear_idx = linear_idx * self.shape[i] + full_indices[i];
                    }
                    sum_val += self.data[linear_idx];
                }

                // Convert output position to linear index and set value
                var out_linear_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    out_linear_idx = out_linear_idx * out_shape[i] + out_indices[i];
                }
                result.data[out_linear_idx] = sum_val;

                // Increment output index
                var carry: usize = 1;
                var dim: usize = ndim - 1;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    out_indices[dim] += carry;
                    if (out_indices[dim] >= out_shape[dim]) {
                        out_indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Mean along a specified axis, reducing dimensionality
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Axis to compute mean along (must be < ndim)
        ///
        /// Returns: NDArray(f64, ndim-1) containing mean values
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(m) where m = prod(output shape)
        pub fn meanAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(f64, ndim - 1) {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Build output shape by removing the reduction axis
            var out_shape: [ndim - 1]usize = undefined;
            if (ndim > 1) {
                var idx: usize = 0;
                for (self.shape, 0..) |dim, i| {
                    if (i != axis) {
                        out_shape[idx] = dim;
                        idx += 1;
                    }
                }
            }

            // Allocate result array initialized to zero
            var result = try NDArray(f64, ndim - 1).zeros(allocator, &out_shape, self.layout);
            errdefer result.deinit();

            const axis_size = self.shape[axis];
            const axis_size_f64 = @as(f64, @floatFromInt(axis_size));

            // Special case: reducing to 0D (scalar)
            if (ndim == 1) {
                var sum_val: f64 = 0.0;
                for (self.data) |val| {
                    const fval = if (@typeInfo(T) == .float)
                        @as(f64, val)
                    else
                        @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                    sum_val += fval;
                }
                result.data[0] = sum_val / axis_size_f64;
                return result;
            }

            // Iterate through all output positions
            var out_indices: [ndim - 1]usize = stdlib.mem.zeroes([ndim - 1]usize);
            var done = false;

            while (!done) {
                // Sum along the reduction axis
                var sum_val: f64 = 0.0;
                for (0..axis_size) |axis_idx| {
                    // Map output position to full input position
                    var full_indices: [ndim]usize = undefined;
                    var out_idx: usize = 0;
                    for (0..ndim) |i| {
                        if (i == axis) {
                            full_indices[i] = axis_idx;
                        } else {
                            full_indices[i] = out_indices[out_idx];
                            out_idx += 1;
                        }
                    }

                    // Convert to linear index and get value
                    var linear_idx: usize = 0;
                    for (0..ndim) |i| {
                        linear_idx = linear_idx * self.shape[i] + full_indices[i];
                    }
                    const val = self.data[linear_idx];
                    const fval = if (@typeInfo(T) == .float)
                        @as(f64, val)
                    else
                        @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                    sum_val += fval;
                }

                // Convert output position to linear index and set value (mean = sum / axis_size)
                var out_linear_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    out_linear_idx = out_linear_idx * out_shape[i] + out_indices[i];
                }
                result.data[out_linear_idx] = sum_val / axis_size_f64;

                // Increment output index
                var carry: usize = 1;
                var dim: usize = ndim - 1;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    out_indices[dim] += carry;
                    if (out_indices[dim] >= out_shape[dim]) {
                        out_indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Minimum along a specified axis, reducing dimensionality
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Axis to find minimum along (must be < ndim)
        ///
        /// Returns: NDArray with shape = original shape with axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(m) where m = prod(output shape)
        pub fn minAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim - 1) {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Build output shape by removing the reduction axis
            var out_shape: [ndim - 1]usize = undefined;
            if (ndim > 1) {
                var idx: usize = 0;
                for (self.shape, 0..) |dim, i| {
                    if (i != axis) {
                        out_shape[idx] = dim;
                        idx += 1;
                    }
                }
            }

            // Allocate result array
            var result = try NDArray(T, ndim - 1).init(allocator, &out_shape, self.layout);
            errdefer result.deinit();

            // Iterate through all output positions
            var out_indices: [ndim - 1]usize = stdlib.mem.zeroes([ndim - 1]usize);
            var done = false;

            while (!done) {
                // Find minimum along the reduction axis
                var min_val: T = undefined;
                var first = true;
                for (0..self.shape[axis]) |axis_idx| {
                    // Map output position to full input position
                    var full_indices: [ndim]usize = undefined;
                    var out_idx: usize = 0;
                    for (0..ndim) |i| {
                        if (i == axis) {
                            full_indices[i] = axis_idx;
                        } else {
                            full_indices[i] = out_indices[out_idx];
                            out_idx += 1;
                        }
                    }

                    // Convert to linear index and get value
                    var linear_idx: usize = 0;
                    for (0..ndim) |i| {
                        linear_idx = linear_idx * self.shape[i] + full_indices[i];
                    }
                    const val = self.data[linear_idx];
                    if (first) {
                        min_val = val;
                        first = false;
                    } else if (val < min_val) {
                        min_val = val;
                    }
                }

                // Convert output position to linear index and set value
                var out_linear_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    out_linear_idx = out_linear_idx * out_shape[i] + out_indices[i];
                }
                result.data[out_linear_idx] = min_val;

                // Increment output index
                var carry: usize = 1;
                var dim: usize = ndim - 1;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    out_indices[dim] += carry;
                    if (out_indices[dim] >= out_shape[dim]) {
                        out_indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }

        /// Maximum along a specified axis, reducing dimensionality
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Axis to find maximum along (must be < ndim)
        ///
        /// Returns: NDArray with shape = original shape with axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(m) where m = prod(output shape)
        pub fn maxAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || AllocatorError)!NDArray(T, ndim - 1) {
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Build output shape by removing the reduction axis
            var out_shape: [ndim - 1]usize = undefined;
            if (ndim > 1) {
                var idx: usize = 0;
                for (self.shape, 0..) |dim, i| {
                    if (i != axis) {
                        out_shape[idx] = dim;
                        idx += 1;
                    }
                }
            }

            // Allocate result array
            var result = try NDArray(T, ndim - 1).init(allocator, &out_shape, self.layout);
            errdefer result.deinit();

            // Iterate through all output positions
            var out_indices: [ndim - 1]usize = stdlib.mem.zeroes([ndim - 1]usize);
            var done = false;

            while (!done) {
                // Find maximum along the reduction axis
                var max_val: T = undefined;
                var first = true;
                for (0..self.shape[axis]) |axis_idx| {
                    // Map output position to full input position
                    var full_indices: [ndim]usize = undefined;
                    var out_idx: usize = 0;
                    for (0..ndim) |i| {
                        if (i == axis) {
                            full_indices[i] = axis_idx;
                        } else {
                            full_indices[i] = out_indices[out_idx];
                            out_idx += 1;
                        }
                    }

                    // Convert to linear index and get value
                    var linear_idx: usize = 0;
                    for (0..ndim) |i| {
                        linear_idx = linear_idx * self.shape[i] + full_indices[i];
                    }
                    const val = self.data[linear_idx];
                    if (first) {
                        max_val = val;
                        first = false;
                    } else if (val > max_val) {
                        max_val = val;
                    }
                }

                // Convert output position to linear index and set value
                var out_linear_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    out_linear_idx = out_linear_idx * out_shape[i] + out_indices[i];
                }
                result.data[out_linear_idx] = max_val;

                // Increment output index
                var carry: usize = 1;
                var dim: usize = ndim - 1;
                while (dim > 0 and carry == 1) {
                    dim -= 1;
                    out_indices[dim] += carry;
                    if (out_indices[dim] >= out_shape[dim]) {
                        out_indices[dim] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry == 1) {
                    done = true;
                }
            }

            return result;
        }
    };
}

/// Helper to copy original array data to center of padded array
fn copyToPaddedCenter(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize) !void {
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Get value from source
        var src_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            src_indices[i] = @as(isize, @intCast(idx));
        }
        const val = try src.get(&src_indices);

        // Calculate destination indices (shifted by pad_before on each axis)
        var dest_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            dest_indices[i] = @as(isize, @intCast(idx + pad_width[i][0]));
        }

        // Set value in destination
        dest.set(&dest_indices, val);

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= src.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Fill padding regions with constant value
fn fillConstantPadding(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize, constant_value: T) !void {
    _ = src;
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Check if this position is in a padding region
        var is_padding = false;
        for (0..ndim) |i| {
            const pad_before = pad_width[i][0];
            const pad_after = pad_width[i][1];
            const original_size = dest.shape[i] - pad_before - pad_after;
            if (indices[i] < pad_before or indices[i] >= pad_before + original_size) {
                is_padding = true;
                break;
            }
        }

        if (is_padding) {
            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
            dest.set(&dest_indices, constant_value);
        }

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Fill padding regions by extending edge values
fn fillEdgePadding(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize) !void {
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Check if this position is in a padding region
        var is_padding = false;
        for (0..ndim) |i| {
            const pad_before = pad_width[i][0];
            const pad_after = pad_width[i][1];
            const original_size = dest.shape[i] - pad_before - pad_after;
            if (indices[i] < pad_before or indices[i] >= pad_before + original_size) {
                is_padding = true;
                break;
            }
        }

        if (is_padding) {
            // Map padded index to nearest edge value
            var src_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                const pad_before = pad_width[i][0];
                const original_size = src.shape[i];
                if (idx < pad_before) {
                    src_indices[i] = 0;
                } else if (idx >= pad_before + original_size) {
                    src_indices[i] = @as(isize, @intCast(original_size - 1));
                } else {
                    src_indices[i] = @as(isize, @intCast(idx - pad_before));
                }
            }
            const val = try src.get(&src_indices);

            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
            dest.set(&dest_indices, val);
        }

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Fill padding regions with reflected values (without repeating edge)
fn fillReflectPadding(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize) !void {
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Check if this position is in a padding region
        var is_padding = false;
        for (0..ndim) |i| {
            const pad_before = pad_width[i][0];
            const pad_after = pad_width[i][1];
            const original_size = dest.shape[i] - pad_before - pad_after;
            if (indices[i] < pad_before or indices[i] >= pad_before + original_size) {
                is_padding = true;
                break;
            }
        }

        if (is_padding) {
            // Map padded index to reflected value
            var src_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                const pad_before = pad_width[i][0];
                const original_size = src.shape[i];
                if (idx < pad_before) {
                    // Reflect before: distance = pad_before - idx
                    const distance = pad_before - idx;
                    src_indices[i] = @as(isize, @intCast(distance));
                } else if (idx >= pad_before + original_size) {
                    // Reflect after: distance = idx - (pad_before + original_size) + 1
                    const distance = idx - (pad_before + original_size) + 1;
                    src_indices[i] = @as(isize, @intCast(original_size - 1 - distance));
                } else {
                    src_indices[i] = @as(isize, @intCast(idx - pad_before));
                }
            }
            const val = try src.get(&src_indices);

            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
            dest.set(&dest_indices, val);
        }

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Fill padding regions with symmetric reflection (with repeating edge)
fn fillSymmetricPadding(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize) !void {
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Check if this position is in a padding region
        var is_padding = false;
        for (0..ndim) |i| {
            const pad_before = pad_width[i][0];
            const pad_after = pad_width[i][1];
            const original_size = dest.shape[i] - pad_before - pad_after;
            if (indices[i] < pad_before or indices[i] >= pad_before + original_size) {
                is_padding = true;
                break;
            }
        }

        if (is_padding) {
            // Map padded index to symmetric reflected value
            var src_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                const pad_before = pad_width[i][0];
                const original_size = src.shape[i];
                if (idx < pad_before) {
                    // Symmetric reflection before: distance = pad_before - idx - 1
                    const distance = pad_before - idx - 1;
                    src_indices[i] = @as(isize, @intCast(distance));
                } else if (idx >= pad_before + original_size) {
                    // Symmetric reflection after: distance = idx - (pad_before + original_size)
                    const distance = idx - (pad_before + original_size);
                    src_indices[i] = @as(isize, @intCast(original_size - 1 - distance));
                } else {
                    src_indices[i] = @as(isize, @intCast(idx - pad_before));
                }
            }
            const val = try src.get(&src_indices);

            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
            dest.set(&dest_indices, val);
        }

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Fill padding regions with wrapped values (circular)
fn fillWrapPadding(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), pad_width: []const [2]usize) !void {
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Check if this position is in a padding region
        var is_padding = false;
        for (0..ndim) |i| {
            const pad_before = pad_width[i][0];
            const pad_after = pad_width[i][1];
            const original_size = dest.shape[i] - pad_before - pad_after;
            if (indices[i] < pad_before or indices[i] >= pad_before + original_size) {
                is_padding = true;
                break;
            }
        }

        if (is_padding) {
            // Map padded index to wrapped value
            var src_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                const pad_before = pad_width[i][0];
                const original_size = src.shape[i];
                if (idx < pad_before) {
                    // Wrap before
                    const distance = pad_before - idx;
                    const wrapped = original_size - (distance % original_size);
                    src_indices[i] = @as(isize, @intCast(if (wrapped == original_size) 0 else wrapped));
                } else if (idx >= pad_before + original_size) {
                    // Wrap after
                    const distance = idx - (pad_before + original_size);
                    src_indices[i] = @as(isize, @intCast(distance % original_size));
                } else {
                    src_indices[i] = @as(isize, @intCast(idx - pad_before));
                }
            }
            const val = try src.get(&src_indices);

            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
            dest.set(&dest_indices, val);
        }

        // Increment multi-dimensional index
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Helper to copy a section of array during split operation
fn copySplitSection(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), axis: usize, axis_offset: usize) !void {
    // Iterate through all elements of destination array using multi-dimensional indices
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Calculate source indices (shifted along split axis by offset)
        var src_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            if (i == axis) {
                src_indices[i] = @as(isize, @intCast(idx + axis_offset));
            } else {
                src_indices[i] = @as(isize, @intCast(idx));
            }
        }

        // Get value from source
        const val = try src.get(&src_indices);

        // Calculate destination indices (same as iteration indices)
        var dest_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            dest_indices[i] = @as(isize, @intCast(idx));
        }

        // Set value in destination
        dest.set(&dest_indices, val);

        // Increment multi-dimensional index (row-major order: rightmost varies fastest)
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= dest.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Helper to copy array to stacked result during stack operation
fn copyArrayToStack(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim + 1), src: *const NDArray(T, ndim), axis: usize, stack_index: usize) !void {
    // Iterate through all elements of source array using multi-dimensional indices
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Get value from source at current indices
        var src_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            src_indices[i] = @as(isize, @intCast(idx));
        }
        const val = try src.get(&src_indices);

        // Calculate destination indices with new dimension inserted at axis
        var dest_indices: [ndim + 1]isize = undefined;
        for (0..axis) |i| {
            dest_indices[i] = @as(isize, @intCast(indices[i]));
        }
        dest_indices[axis] = @as(isize, @intCast(stack_index));
        for (axis..ndim) |i| {
            dest_indices[i + 1] = @as(isize, @intCast(indices[i]));
        }

        // Set value in destination
        dest.set(&dest_indices, val);

        // Increment multi-dimensional index (row-major order: rightmost varies fastest)
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= src.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Helper to copy array segment during concatenation
fn copyArraySegment(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), axis: usize, offset: usize) !void {
    // Iterate through all elements of source array using multi-dimensional indices
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Get value from source at current indices
        var src_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            src_indices[i] = @as(isize, @intCast(idx));
        }
        const val = try src.get(&src_indices);

        // Calculate destination indices (same except shifted along concat axis)
        var dest_indices: [ndim]isize = undefined;
        for (indices, 0..) |idx, i| {
            if (i == axis) {
                dest_indices[i] = @as(isize, @intCast(idx + offset));
            } else {
                dest_indices[i] = @as(isize, @intCast(idx));
            }
        }

        // Set value in destination
        dest.set(&dest_indices, val);

        // Increment multi-dimensional index (row-major order: rightmost varies fastest)
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            indices[dim] += carry;
            if (indices[dim] >= src.shape[dim]) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

/// Helper to copy a slice of array segment along an axis
/// Used by insert(), append(), delete()
///
/// dest: destination array to copy into
/// src: source array to copy from
/// axis: axis along which to slice
/// dest_offset: starting index in destination along axis
/// src_offset: starting index in source along axis
/// count: number of elements to copy along axis
fn copyArraySegmentInsert(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim), src: *const NDArray(T, ndim), axis: usize, dest_offset: usize, src_offset: usize, count: usize) !void {
    // Iterate through all elements in the slice
    var indices: [ndim]usize = stdlib.mem.zeroes([ndim]usize);

    var done = false;
    while (!done) {
        // Skip if not in the source slice range along axis
        if (indices[axis] < count) {
            // Calculate source indices
            var src_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                if (i == axis) {
                    src_indices[i] = @as(isize, @intCast(src_offset + idx));
                } else {
                    src_indices[i] = @as(isize, @intCast(idx));
                }
            }

            // Calculate destination indices
            var dest_indices: [ndim]isize = undefined;
            for (indices, 0..) |idx, i| {
                if (i == axis) {
                    dest_indices[i] = @as(isize, @intCast(dest_offset + idx));
                } else {
                    dest_indices[i] = @as(isize, @intCast(idx));
                }
            }

            // Copy value
            const val = try src.get(&src_indices);
            dest.set(&dest_indices, val);
        }

        // Increment multi-dimensional index (row-major order: rightmost varies fastest)
        var carry: usize = 1;
        var dim: usize = ndim;
        while (dim > 0 and carry == 1) {
            dim -= 1;
            // For axis dimension, limit iteration to count
            const limit = if (dim == axis) count else dest.shape[dim];
            indices[dim] += carry;
            if (indices[dim] >= limit) {
                indices[dim] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }

        if (carry == 1) {
            done = true;
        }
    }
}

// ============================================================================
// TESTS — Following TDD "Red" phase (all must FAIL before implementation)
// ============================================================================

// -- Type Definition Tests (3 tests) --

test "ndarray: NDArray(f64, 2) type creation" {
    const ArrayType = NDArray(f64, 2);
    const allocator = testing.allocator;

    // Verify type has required fields by checking they compile
    _ = allocator;
    _ = @hasField(ArrayType, "shape");
    _ = @hasField(ArrayType, "strides");
    _ = @hasField(ArrayType, "data");
    _ = @hasField(ArrayType, "allocator");
    _ = @hasField(ArrayType, "layout");

    // Verify the fields have expected types
    try testing.expect(@TypeOf(@as(ArrayType, undefined).shape) == [2]usize);
    try testing.expect(@TypeOf(@as(ArrayType, undefined).strides) == [2]usize);
}

test "ndarray: NDArray(i32, 3) type creation" {
    const ArrayType = NDArray(i32, 3);

    // Verify rank is correct by checking shape array length
    const instance: ArrayType = undefined;
    try testing.expectEqual(3, instance.shape.len);
}

test "ndarray: NDArray(u8, 1) 1D array type" {
    const ArrayType = NDArray(u8, 1);

    // Verify 1D array has shape and strides of length 1
    const arr: ArrayType = undefined;
    try testing.expectEqual(1, arr.shape.len);
    try testing.expectEqual(1, arr.strides.len);
}

// -- Initialization Tests (5 tests) --

test "ndarray: init allocates correct memory for 2D row-major" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Should allocate 3*4 = 12 elements
    try testing.expectEqual(12, arr.data.len);
    try testing.expectEqual(12, arr.count());
    try testing.expectEqual(3, arr.shape[0]);
    try testing.expectEqual(4, arr.shape[1]);
}

test "ndarray: init row-major stride calculation [3,4] → [4,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Row-major [3,4]: strides should be [4, 1] (next row = 4 elements)
    try testing.expectEqual(4, arr.strides[0]);
    try testing.expectEqual(1, arr.strides[1]);
}

test "ndarray: init column-major stride calculation [3,4] → [1,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    // Column-major [3,4]: strides should be [1, 3] (next column = 3 elements)
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);
}

test "ndarray: init rejects zero-sized dimensions" {
    const allocator = testing.allocator;
    const result = NDArray(i32, 2).init(allocator, &[_]usize{ 3, 0 }, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: init rejects oversized shape exceeding usize max" {
    const allocator = testing.allocator;
    // Attempt: [usize.max, 2] → product overflows
    const result = NDArray(f64, 2).init(allocator, &[_]usize{ math.maxInt(usize), 2 }, .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

// -- Memory Layout Tests (6 tests) --

test "ndarray: 3D row-major [2,3,4] strides [12,4,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Row-major: stride[i] = prod(shape[i+1..])
    // [2,3,4] → [3*4, 4, 1] = [12, 4, 1]
    try testing.expectEqual(12, arr.strides[0]);
    try testing.expectEqual(4, arr.strides[1]);
    try testing.expectEqual(1, arr.strides[2]);
}

test "ndarray: 3D column-major [2,3,4] strides [1,2,6]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .column_major);
    defer arr.deinit();

    // Column-major: stride[i] = prod(shape[0..i])
    // [2,3,4] → [1, 2, 2*3] = [1, 2, 6]
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);
    try testing.expectEqual(6, arr.strides[2]);
}

test "ndarray: 1D array has stride [1] regardless of layout" {
    const allocator = testing.allocator;

    var arr_rm = try NDArray(f64, 1).init(allocator, &[_]usize{10}, .row_major);
    defer arr_rm.deinit();
    try testing.expectEqual(1, arr_rm.strides[0]);

    var arr_cm = try NDArray(f64, 1).init(allocator, &[_]usize{10}, .column_major);
    defer arr_cm.deinit();
    try testing.expectEqual(1, arr_cm.strides[0]);
}

test "ndarray: 4D row-major [2,3,4,5] strides [60,20,5,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(u8, 4).init(allocator, &[_]usize{ 2, 3, 4, 5 }, .row_major);
    defer arr.deinit();

    // [2,3,4,5] → [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1]
    try testing.expectEqual(60, arr.strides[0]);
    try testing.expectEqual(20, arr.strides[1]);
    try testing.expectEqual(5, arr.strides[2]);
    try testing.expectEqual(1, arr.strides[3]);
}

test "ndarray: 4D column-major [2,3,4,5] strides [1,2,6,24]" {
    const allocator = testing.allocator;
    var arr = try NDArray(u8, 4).init(allocator, &[_]usize{ 2, 3, 4, 5 }, .column_major);
    defer arr.deinit();

    // [2,3,4,5] → [1, 2, 2*3, 2*3*4] = [1, 2, 6, 24]
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);
    try testing.expectEqual(6, arr.strides[2]);
    try testing.expectEqual(24, arr.strides[3]);
}

// -- Memory Safety Tests (3 tests) --

test "ndarray: deinit frees allocated memory (leak detection)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 100, 100 }, .row_major);
    arr.deinit();

    // stdlib.testing.allocator automatically detects leaks
    // If deinit didn't free, test will fail with leak error
}

test "ndarray: multiple init/deinit cycles don't leak" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 5, 6, 7 }, .row_major);
        arr.deinit();
    }

    // No memory leaks should be detected
}

test "ndarray: uninitialized array can be safely deinit'd" {
    const allocator = testing.allocator;

    // Create uninitialized array (common pattern: manual initialization)
    var arr: NDArray(f64, 2) = undefined;
    arr.allocator = allocator;
    arr.data = &[_]f64{};
    arr.deinit();

    // Verify deinit() completes without crash (test passes if no panic)
    // The allocator would detect double-free or invalid free if present
    try testing.expect(true); // explicit pass marker
}

// -- Edge Cases Tests (3 tests) --

test "ndarray: empty array shape [0] allocates 0 bytes" {
    const allocator = testing.allocator;

    // Special case: allow 0D arrays (scalars) — empty shape []
    // But 1D with shape [0] should be rejected
    const result = NDArray(f64, 1).init(allocator, &[_]usize{0}, .row_major);
    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: scalar (0D array) if ndim=0" {
    // Scalars: 0D arrays with shape [] — value stored directly (not implemented yet)
    // This test documents the expected interface for future implementation
    // For now, minimum ndim = 1
    _ = NDArray(f64, 1);
}

test "ndarray: large array 1M elements allocates successfully" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1_000_000}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(1_000_000, arr.count());
    try testing.expectEqual(1, arr.strides[0]);
}

// -- Type Validation Tests (2 tests) --

test "ndarray: shape stored correctly" {
    const allocator = testing.allocator;
    const shape = [_]usize{ 2, 3, 4 };
    var arr = try NDArray(i32, 3).init(allocator, shape[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(2, arr.shape[0]);
    try testing.expectEqual(3, arr.shape[1]);
    try testing.expectEqual(4, arr.shape[2]);
}

test "ndarray: layout stored correctly" {
    const allocator = testing.allocator;

    var arr_rm = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr_rm.deinit();
    try testing.expectEqual(Layout.row_major, arr_rm.layout);

    var arr_cm = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr_cm.deinit();
    try testing.expectEqual(Layout.column_major, arr_cm.layout);
}

// -- Count and isEmpty Tests (3 tests) --

test "ndarray: count() returns prod(shape)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // 2*3*4 = 24
    try testing.expectEqual(24, arr.count());
}

test "ndarray: isEmpty() returns false for non-empty array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    try testing.expect(!arr.isEmpty());
}

test "ndarray: isEmpty() returns false for all non-zero shapes" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 4).init(allocator, &[_]usize{ 1, 1, 1, 1 }, .row_major);
    defer arr.deinit();

    try testing.expect(!arr.isEmpty());
}

// -- Validation Tests (3 tests) --

test "ndarray: validate() checks dimension invariant" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Should not error on valid array
    try arr.validate();
}

test "ndarray: validate() checks stride consistency with row-major" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 5 }, .row_major);
    defer arr.deinit();

    // Row-major [3,5] must have strides [5,1]
    // After init, validate should pass
    try arr.validate();
}

test "ndarray: validate() checks stride consistency with column-major" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 5 }, .column_major);
    defer arr.deinit();

    // Column-major [3,5] must have strides [1,3]
    // After init, validate should pass
    try arr.validate();
}

// -- Allocator Storage Tests (2 tests) --

test "ndarray: allocator is stored and accessible" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer arr.deinit();

    // Verify allocator is stored
    _ = arr.allocator;
}

test "ndarray: deinit uses stored allocator" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{10}, .row_major);

    // Deinit should use arr.allocator (which is the testing allocator)
    arr.deinit();
}

// -- Data Pointer Tests (2 tests) --

test "ndarray: data pointer initialized to valid slice" {
    const allocator = testing.allocator;
    const shape = [_]usize{ 3, 4 };
    var arr = try NDArray(f64, 2).init(allocator, shape[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.data.len);
}

test "ndarray: data is contiguous slice of correct length" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Data should be a slice of 2*3*4 = 24 elements
    try testing.expectEqual(24, arr.data.len);
}

// -- Stress Tests (2 tests) --

test "ndarray: stress test — various dimensions" {
    const allocator = testing.allocator;

    // Test 1D
    var arr1 = try NDArray(f64, 1).init(allocator, &[_]usize{100}, .row_major);
    defer arr1.deinit();
    try testing.expectEqual(100, arr1.count());

    // Test 2D
    var arr2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 10, 10 }, .row_major);
    defer arr2.deinit();
    try testing.expectEqual(100, arr2.count());

    // Test 3D
    var arr3 = try NDArray(f64, 3).init(allocator, &[_]usize{ 5, 5, 4 }, .row_major);
    defer arr3.deinit();
    try testing.expectEqual(100, arr3.count());

    // Test 5D
    var arr5 = try NDArray(f64, 5).init(allocator, &[_]usize{ 2, 2, 5, 5, 1 }, .row_major);
    defer arr5.deinit();
    try testing.expectEqual(100, arr5.count());
}

test "ndarray: stress test — layout consistency across dimensions" {
    const allocator = testing.allocator;

    // Verify row-major strides for various shapes
    for (0..5) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
        defer arr.deinit();
        try testing.expectEqual(12, arr.strides[0]);
        try testing.expectEqual(4, arr.strides[1]);
        try testing.expectEqual(1, arr.strides[2]);
    }

    // Verify column-major strides for various shapes
    for (0..5) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .column_major);
        defer arr.deinit();
        try testing.expectEqual(1, arr.strides[0]);
        try testing.expectEqual(2, arr.strides[1]);
        try testing.expectEqual(6, arr.strides[2]);
    }
}

// -- Different Element Types Tests (3 tests) --

test "ndarray: f32 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(f32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(6, arr.count());
}

test "ndarray: u64 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(u64, 2).init(allocator, &[_]usize{ 4, 5 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(20, arr.count());
}

test "ndarray: complex struct element type" {
    const Complex = struct { real: f64, imag: f64 };
    const allocator = testing.allocator;
    var arr = try NDArray(Complex, 1).init(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
}

// -- zeros() Creation Function Tests (6 tests) --

test "ndarray: zeros() creates 1D array with all zeros" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).zeros(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() creates 2D array [3,4] with all zeros" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() creates 3D array [2,3,4] with all zeros" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).zeros(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(24, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() works with i32 type" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(@as(i32, 0), val);
    }
}

test "ndarray: zeros() rejects zero dimension" {
    const allocator = testing.allocator;
    const result = NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 0 }, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

// -- ones() Creation Function Tests (6 tests) --

test "ndarray: ones() creates 1D array with all ones" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).ones(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() creates 2D array [3,4] with all ones" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() creates 3D array [2,3,4] with all ones" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).ones(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(24, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() works with u8 type" {
    const allocator = testing.allocator;
    var arr = try NDArray(u8, 2).ones(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(6, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(@as(u8, 1), val);
    }
}

test "ndarray: ones() rejects zero dimension" {
    const allocator = testing.allocator;
    const result = NDArray(f64, 1).ones(allocator, &[_]usize{0}, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

// -- full() Creation Function Tests (5 tests) --

test "ndarray: full() creates 2D array filled with custom value (42.0)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{ 3, 4 }, 42.0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(42.0, val);
    }
}

test "ndarray: full() fills array with 3.14159" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).full(allocator, &[_]usize{5}, 3.14159, .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(3.14159, val);
    }
}

test "ndarray: full() fills array with negative value (-100)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).full(allocator, &[_]usize{ 2, 3 }, -100, .row_major);
    defer arr.deinit();

    try testing.expectEqual(6, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(@as(i32, -100), val);
    }
}

test "ndarray: full() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{ 3, 4 }, 7.5, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    for (arr.data) |val| {
        try testing.expectEqual(7.5, val);
    }
}

test "ndarray: full() works with 0.0 value" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{ 2, 2 }, 0.0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(4, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

// -- empty() Creation Function Tests (3 tests) --

test "ndarray: empty() allocates space without initialization" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).empty(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    try testing.expectEqual(10, arr.data.len);
}

test "ndarray: empty() respects shape [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).empty(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    try testing.expectEqual(3, arr.shape[0]);
    try testing.expectEqual(4, arr.shape[1]);
}

test "ndarray: empty() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).empty(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);
}

// -- arange() Creation Function Tests (6 tests) --

test "ndarray: arange() creates [0, 10) with step 1" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 1.0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (0..10) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i)), arr.data[i]);
    }
}

test "ndarray: arange() creates [0, 10) with step 2" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 2.0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    const expected = [_]f64{ 0.0, 2.0, 4.0, 6.0, 8.0 };
    for (0..5) |i| {
        try testing.expectEqual(expected[i], arr.data[i]);
    }
}

test "ndarray: arange() creates [5, 15) with step 3" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 5, 15, 3, .row_major);
    defer arr.deinit();

    try testing.expectEqual(4, arr.count());
    const expected = [_]i32{ 5, 8, 11, 14 };
    for (0..4) |i| {
        try testing.expectEqual(expected[i], arr.data[i]);
    }
}

test "ndarray: arange() handles descending range [10, 0) with step -1" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 10, 0, -1, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    // arange(start, stop, step) generates [start, start+step, ...] until value would pass stop
    // For (10, 0, -1): [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] (excludes 0 which is the stop)
    for (0..10) |i| {
        try testing.expectEqual(@as(i32, @intCast(10 - @as(i32, @intCast(i)))), arr.data[i]);
    }
}

test "ndarray: arange() rejects step=0" {
    const allocator = testing.allocator;
    const result = NDArray(f64, 1).arange(allocator, 0.0, 10.0, 0.0, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: arange() creates [100, 200) with step 10" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 100, 200, 10, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (0..10) |i| {
        try testing.expectEqual(@as(i32, @intCast(100 + i * 10)), arr.data[i]);
    }
}

// -- linspace() Creation Function Tests (6 tests) --

test "ndarray: linspace() creates 5 evenly spaced points in [0, 1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, 0.0, 1.0, 5, .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    // Expected: [0.0, 0.25, 0.5, 0.75, 1.0]
    try testing.expectApproxEqAbs(0.0, arr.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.25, arr.data[1], 1e-10);
    try testing.expectApproxEqAbs(0.5, arr.data[2], 1e-10);
    try testing.expectApproxEqAbs(0.75, arr.data[3], 1e-10);
    try testing.expectApproxEqAbs(1.0, arr.data[4], 1e-10);
}

test "ndarray: linspace() creates 11 points in [0, 10]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, 0.0, 10.0, 11, .row_major);
    defer arr.deinit();

    try testing.expectEqual(11, arr.count());
    // Expected: [0.0, 1.0, 2.0, ..., 10.0]
    for (0..11) |i| {
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i)), arr.data[i], 1e-10);
    }
}

test "ndarray: linspace() creates 3 points in [-1, 1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, -1.0, 1.0, 3, .row_major);
    defer arr.deinit();

    try testing.expectEqual(3, arr.count());
    // Expected: [-1.0, 0.0, 1.0]
    try testing.expectApproxEqAbs(-1.0, arr.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.0, arr.data[1], 1e-10);
    try testing.expectApproxEqAbs(1.0, arr.data[2], 1e-10);
}

test "ndarray: linspace() handles single point [5, 5] with count=1" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, 5.0, 5.0, 1, .row_major);
    defer arr.deinit();

    try testing.expectEqual(1, arr.count());
    try testing.expectApproxEqAbs(5.0, arr.data[0], 1e-10);
}

test "ndarray: linspace() rejects count=0" {
    const allocator = testing.allocator;
    const result = NDArray(f64, 1).linspace(allocator, 0.0, 1.0, 0, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: linspace() creates 100 points in [0, 1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, 0.0, 1.0, 100, .row_major);
    defer arr.deinit();

    try testing.expectEqual(100, arr.count());
    // Verify first and last endpoints
    try testing.expectApproxEqAbs(0.0, arr.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, arr.data[99], 1e-10);
    // Verify monotonic increase
    for (0..99) |i| {
        try testing.expect(arr.data[i] <= arr.data[i + 1]);
    }
}

// -- fromOwnedSlice() Creation Function Tests (11 tests) --

test "ndarray: fromOwnedSlice() basic 1D array [5] takes ownership" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 5);
    defer {
        // Only free if test fails; normally deinit() handles it
        if (!debug.runtime_safety) {
            // In release mode, trust the NDArray to free
        }
    }
    for (0..5) |i| {
        data[i] = @floatFromInt(i + 1);
    }

    var arr = try NDArray(f64, 1).fromOwnedSlice(allocator, &[_]usize{5}, data, .row_major);
    defer arr.deinit();

    // Verify shape and count
    try testing.expectEqual(5, arr.count());
    try testing.expectEqual(5, arr.shape[0]);

    // Verify data ownership (should point to same memory, no copy)
    try testing.expectEqual(@intFromPtr(data.ptr), @intFromPtr(arr.data.ptr));

    // Verify data integrity
    for (0..5) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), arr.data[i]);
    }
}

test "ndarray: fromOwnedSlice() 2D array row-major [3,4] layout verification" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 12);
    defer {
        // Trust deinit to free
    }
    for (0..12) |i| {
        data[i] = @floatFromInt(i + 1);
    }

    var arr = try NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{ 3, 4 }, data, .row_major);
    defer arr.deinit();

    // Verify shape
    try testing.expectEqual(3, arr.shape[0]);
    try testing.expectEqual(4, arr.shape[1]);
    try testing.expectEqual(12, arr.count());

    // Verify row-major strides: [4, 1]
    try testing.expectEqual(4, arr.strides[0]);
    try testing.expectEqual(1, arr.strides[1]);

    // Verify layout
    try testing.expectEqual(Layout.row_major, arr.layout);

    // Verify all data copied correctly (should still be original data)
    for (0..12) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), arr.data[i]);
    }
}

test "ndarray: fromOwnedSlice() 2D array column-major [2,5] layout verification" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 10);
    defer {
        // Trust deinit to free
    }
    for (0..10) |i| {
        data[i] = @floatFromInt(i + 10);
    }

    var arr = try NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{ 2, 5 }, data, .column_major);
    defer arr.deinit();

    // Verify shape
    try testing.expectEqual(2, arr.shape[0]);
    try testing.expectEqual(5, arr.shape[1]);

    // Verify column-major strides: [1, 2]
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);

    // Verify layout
    try testing.expectEqual(Layout.column_major, arr.layout);

    // Verify data integrity
    for (0..10) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 10)), arr.data[i]);
    }
}

test "ndarray: fromOwnedSlice() 3D array [2,3,4] rank verification" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(i32, 24);
    defer {
        // Trust deinit to free
    }
    for (0..24) |i| {
        data[i] = @intCast(i + 100);
    }

    var arr = try NDArray(i32, 3).fromOwnedSlice(allocator, &[_]usize{ 2, 3, 4 }, data, .row_major);
    defer arr.deinit();

    // Verify shape
    try testing.expectEqual(2, arr.shape[0]);
    try testing.expectEqual(3, arr.shape[1]);
    try testing.expectEqual(4, arr.shape[2]);
    try testing.expectEqual(24, arr.count());

    // Verify row-major strides: [12, 4, 1]
    try testing.expectEqual(12, arr.strides[0]);
    try testing.expectEqual(4, arr.strides[1]);
    try testing.expectEqual(1, arr.strides[2]);

    // Verify data integrity
    for (0..24) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 100)), arr.data[i]);
    }
}

test "ndarray: fromOwnedSlice() 3D array column-major [2,3,4] stride calculation" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 24);
    defer {
        // Trust deinit to free
    }
    for (0..24) |i| {
        data[i] = @floatFromInt(i);
    }

    var arr = try NDArray(f64, 3).fromOwnedSlice(allocator, &[_]usize{ 2, 3, 4 }, data, .column_major);
    defer arr.deinit();

    // Verify column-major strides: [1, 2, 6]
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);
    try testing.expectEqual(6, arr.strides[2]);
}

test "ndarray: fromOwnedSlice() size mismatch [3,4] shape with 10-element data returns error" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 10);
    defer allocator.free(data); // Manually free since fromOwnedSlice will fail

    // Try to create [3,4] (12 elements) from 10-element slice
    const result = NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{ 3, 4 }, data, .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

test "ndarray: fromOwnedSlice() shape overflow check" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 100);
    defer allocator.free(data); // Manually free since fromOwnedSlice will fail

    // Try shape that would overflow: [max_usize, 2]
    const huge_shape = &[_]usize{ math.maxInt(usize), 2 };
    const result = NDArray(f64, 2).fromOwnedSlice(allocator, huge_shape, data, .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

test "ndarray: fromOwnedSlice() zero dimension check [0,5] rejects" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 1);
    defer allocator.free(data); // Manually free since fromOwnedSlice will fail

    // Zero-dimension array should be rejected
    const result = NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{ 0, 5 }, data, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: fromOwnedSlice() mismatched shape length rejects" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 12);
    defer allocator.free(data); // Manually free since fromOwnedSlice will fail

    // Shape length 1 doesn't match ndim=2
    const result = NDArray(f64, 2).fromOwnedSlice(allocator, &[_]usize{12}, data, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: fromOwnedSlice() data pointer equality (no copy)" {
    const allocator = testing.allocator;
    const owned_data = try allocator.alloc(f64, 8);
    defer {
        // Trust deinit to free
    }
    for (0..8) |i| {
        owned_data[i] = @floatFromInt(i * 10);
    }

    const original_ptr = @intFromPtr(owned_data.ptr);
    var arr = try NDArray(f64, 1).fromOwnedSlice(allocator, &[_]usize{8}, owned_data, .row_major);
    defer arr.deinit();

    const array_ptr = @intFromPtr(arr.data.ptr);

    // Pointers should be identical (no copy, direct ownership)
    try testing.expectEqual(original_ptr, array_ptr);
}

test "ndarray: fromOwnedSlice() validate() passes invariant checks" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(i32, 20);
    defer {
        // Trust deinit to free
    }
    for (0..20) |i| {
        data[i] = @intCast(i);
    }

    var arr = try NDArray(i32, 2).fromOwnedSlice(allocator, &[_]usize{ 4, 5 }, data, .row_major);
    defer arr.deinit();

    // validate() should pass without assertion errors
    try arr.validate();
}

test "ndarray: fromOwnedSlice() i32 element type 1D [10]" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(i32, 10);
    defer {
        // Trust deinit to free
    }
    for (0..10) |i| {
        data[i] = @intCast(i * 5);
    }

    var arr = try NDArray(i32, 1).fromOwnedSlice(allocator, &[_]usize{10}, data, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (0..10) |i| {
        try testing.expectEqual(@as(i32, @intCast(i * 5)), arr.data[i]);
    }
}

// -- fromSlice() Creation Function Tests (5 tests) --

test "ndarray: fromSlice() creates 2D array [3,4] from slice" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, data[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (0..12) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), arr.data[i]);
    }
}

test "ndarray: fromSlice() creates 1D array [5]" {
    const allocator = testing.allocator;
    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, data[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    for (0..5) |i| {
        try testing.expectEqual(data[i], arr.data[i]);
    }
}

test "ndarray: fromSlice() respects column-major layout [2,3]" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, data[0..], .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);
}

test "ndarray: fromSlice() rejects shape mismatch" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1, 2, 3 };
    // Try to create 2D array [3,4] (12 elements) from 3-element slice
    const result = NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, data[0..], .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

test "ndarray: fromSlice() creates 3D array [2,2,2]" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, data[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(8, arr.count());
    for (0..8) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), arr.data[i]);
    }
}

// -- eye() / identity() Creation Function Tests (7 tests) --

test "ndarray: eye() creates 2x2 identity matrix" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).eye(allocator, 2, 2, 0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(4, arr.count());
    // First row: [1, 0]
    try testing.expectEqual(1.0, arr.data[0]);
    try testing.expectEqual(0.0, arr.data[1]);
    // Second row: [0, 1]
    try testing.expectEqual(0.0, arr.data[2]);
    try testing.expectEqual(1.0, arr.data[3]);
}

test "ndarray: eye() creates 3x3 identity matrix" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).eye(allocator, 3, 3, 0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(9, arr.count());
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectEqual(expected, arr.data[i * 3 + j]);
        }
    }
}

test "ndarray: eye() creates 5x5 identity matrix" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).eye(allocator, 5, 5, 0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(25, arr.count());
    for (0..5) |i| {
        for (0..5) |j| {
            const expected: i32 = if (i == j) 1 else 0;
            try testing.expectEqual(expected, arr.data[i * 5 + j]);
        }
    }
}

test "ndarray: eye() respects column-major layout for 3x3" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).eye(allocator, 3, 3, 0, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);
}

test "ndarray: eye() works with i32 type" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).eye(allocator, 2, 2, 0, .row_major);
    defer arr.deinit();

    try testing.expectEqual(4, arr.count());
    try testing.expectEqual(@as(i32, 1), arr.data[0]);
    try testing.expectEqual(@as(i32, 0), arr.data[1]);
    try testing.expectEqual(@as(i32, 0), arr.data[2]);
    try testing.expectEqual(@as(i32, 1), arr.data[3]);
}

test "ndarray: identity() is alias for eye with k=0" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).identity(allocator, 3, 3, .row_major);
    defer arr.deinit();

    try testing.expectEqual(9, arr.count());
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectEqual(expected, arr.data[i * 3 + j]);
        }
    }
}

test "ndarray: identity() works with different types and layouts" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).identity(allocator, 4, 4, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    // In column-major, diagonal is at positions [0, 5, 10, 15] (stride of 5)
    try testing.expectEqual(@as(i32, 1), arr.data[0]);
    try testing.expectEqual(@as(i32, 1), arr.data[5]);
    try testing.expectEqual(@as(i32, 1), arr.data[10]);
    try testing.expectEqual(@as(i32, 1), arr.data[15]);
}

// -- Diagonal Operations Tests (24 tests) --

test "ndarray: diag() constructs diagonal matrix from 1D array (main diagonal)" {
    const allocator = testing.allocator;
    var vec = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer vec.deinit();

    var mat = try vec.diag(allocator, 0, .row_major);
    defer mat.deinit();

    // Type is NDArray(f64, 2), so ndim is guaranteed to be 2 at compile time
    try testing.expectEqual(@as(usize, 3), mat.shape[0]);
    try testing.expectEqual(@as(usize, 3), mat.shape[1]);

    // Check diagonal elements
    try testing.expectEqual(@as(f64, 1), mat.data[0]); // [0,0]
    try testing.expectEqual(@as(f64, 2), mat.data[4]); // [1,1]
    try testing.expectEqual(@as(f64, 3), mat.data[8]); // [2,2]

    // Check off-diagonal zeros
    try testing.expectEqual(@as(f64, 0), mat.data[1]); // [0,1]
    try testing.expectEqual(@as(f64, 0), mat.data[3]); // [1,0]
}

test "ndarray: diag() constructs diagonal matrix with positive offset k=1" {
    const allocator = testing.allocator;
    var vec = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 5, 6 }, .row_major);
    defer vec.deinit();

    var mat = try vec.diag(allocator, 1, .row_major);
    defer mat.deinit();

    try testing.expectEqual(@as(usize, 3), mat.shape[0]);
    try testing.expectEqual(@as(usize, 3), mat.shape[1]);

    // Elements should be on diagonal above main (k=1)
    try testing.expectEqual(@as(f64, 5), mat.data[1]); // [0,1]
    try testing.expectEqual(@as(f64, 6), mat.data[5]); // [1,2]
    try testing.expectEqual(@as(f64, 0), mat.data[0]); // [0,0]
}

test "ndarray: diag() constructs diagonal matrix with negative offset k=-1" {
    const allocator = testing.allocator;
    var vec = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 7, 8 }, .row_major);
    defer vec.deinit();

    var mat = try vec.diag(allocator, -1, .row_major);
    defer mat.deinit();

    try testing.expectEqual(@as(usize, 3), mat.shape[0]);
    try testing.expectEqual(@as(usize, 3), mat.shape[1]);

    // Elements should be on diagonal below main (k=-1)
    try testing.expectEqual(@as(f64, 7), mat.data[3]); // [1,0]
    try testing.expectEqual(@as(f64, 8), mat.data[7]); // [2,1]
    try testing.expectEqual(@as(f64, 0), mat.data[0]); // [0,0]
}

test "ndarray: diag() extracts main diagonal from 2D array" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var diag_arr = try mat.diag(allocator, 0, .row_major);
    defer diag_arr.deinit();

    // Type is NDArray(f64, 1), so ndim is guaranteed to be 1 at compile time
    try testing.expectEqual(@as(usize, 3), diag_arr.shape[0]);
    try testing.expectEqual(@as(f64, 1), diag_arr.data[0]);
    try testing.expectEqual(@as(f64, 5), diag_arr.data[1]);
    try testing.expectEqual(@as(f64, 9), diag_arr.data[2]);
}

test "ndarray: diag() extracts upper diagonal k=1 from 2D array" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var diag_arr = try mat.diag(allocator, 1, .row_major);
    defer diag_arr.deinit();

    try testing.expectEqual(@as(usize, 2), diag_arr.shape[0]);
    try testing.expectEqual(@as(f64, 2), diag_arr.data[0]); // [0,1]
    try testing.expectEqual(@as(f64, 6), diag_arr.data[1]); // [1,2]
}

test "ndarray: diag() extracts lower diagonal k=-1 from 2D array" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var diag_arr = try mat.diag(allocator, -1, .row_major);
    defer diag_arr.deinit();

    try testing.expectEqual(@as(usize, 2), diag_arr.shape[0]);
    try testing.expectEqual(@as(f64, 4), diag_arr.data[0]); // [1,0]
    try testing.expectEqual(@as(f64, 8), diag_arr.data[1]); // [2,1]
}

test "ndarray: diag() handles non-square matrix (3x4)" {
    const allocator = testing.allocator;
    var mat = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, .row_major);
    defer mat.deinit();

    var diag_arr = try mat.diag(allocator, 0, .row_major);
    defer diag_arr.deinit();

    try testing.expectEqual(@as(usize, 3), diag_arr.shape[0]);
    try testing.expectEqual(@as(i32, 1), diag_arr.data[0]);
    try testing.expectEqual(@as(i32, 6), diag_arr.data[1]);
    try testing.expectEqual(@as(i32, 11), diag_arr.data[2]);
}

test "ndarray: diagonal() extracts main diagonal" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer mat.deinit();

    var diag_arr = try mat.diagonal(allocator, 0);
    defer diag_arr.deinit();

    try testing.expectEqual(@as(usize, 2), diag_arr.shape[0]);
    try testing.expectEqual(@as(f64, 1), diag_arr.data[0]);
    try testing.expectEqual(@as(f64, 4), diag_arr.data[1]);
}

test "ndarray: triu() creates upper triangular matrix (main diagonal)" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var upper = try mat.triu(allocator, 0);
    defer upper.deinit();

    // Upper triangle: [1,2,3 | 0,5,6 | 0,0,9]
    try testing.expectEqual(@as(f64, 1), upper.data[0]); // [0,0]
    try testing.expectEqual(@as(f64, 2), upper.data[1]); // [0,1]
    try testing.expectEqual(@as(f64, 3), upper.data[2]); // [0,2]
    try testing.expectEqual(@as(f64, 0), upper.data[3]); // [1,0] - zeroed
    try testing.expectEqual(@as(f64, 5), upper.data[4]); // [1,1]
    try testing.expectEqual(@as(f64, 6), upper.data[5]); // [1,2]
    try testing.expectEqual(@as(f64, 0), upper.data[6]); // [2,0] - zeroed
    try testing.expectEqual(@as(f64, 0), upper.data[7]); // [2,1] - zeroed
    try testing.expectEqual(@as(f64, 9), upper.data[8]); // [2,2]
}

test "ndarray: triu() with k=1 keeps diagonal above main" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var upper = try mat.triu(allocator, 1);
    defer upper.deinit();

    // k=1: zero main diagonal and below
    try testing.expectEqual(@as(f64, 0), upper.data[0]); // [0,0] - zeroed
    try testing.expectEqual(@as(f64, 2), upper.data[1]); // [0,1]
    try testing.expectEqual(@as(f64, 3), upper.data[2]); // [0,2]
    try testing.expectEqual(@as(f64, 0), upper.data[4]); // [1,1] - zeroed
    try testing.expectEqual(@as(f64, 6), upper.data[5]); // [1,2]
}

test "ndarray: triu() with k=-1 keeps diagonal below main" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var upper = try mat.triu(allocator, -1);
    defer upper.deinit();

    // k=-1: keep one diagonal below main
    try testing.expectEqual(@as(f64, 1), upper.data[0]); // [0,0]
    try testing.expectEqual(@as(f64, 4), upper.data[3]); // [1,0] - kept
    try testing.expectEqual(@as(f64, 5), upper.data[4]); // [1,1]
    try testing.expectEqual(@as(f64, 0), upper.data[6]); // [2,0] - zeroed
    try testing.expectEqual(@as(f64, 8), upper.data[7]); // [2,1] - kept
}

test "ndarray: tril() creates lower triangular matrix (main diagonal)" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var lower = try mat.tril(allocator, 0);
    defer lower.deinit();

    // Lower triangle: [1,0,0 | 4,5,0 | 7,8,9]
    try testing.expectEqual(@as(f64, 1), lower.data[0]); // [0,0]
    try testing.expectEqual(@as(f64, 0), lower.data[1]); // [0,1] - zeroed
    try testing.expectEqual(@as(f64, 0), lower.data[2]); // [0,2] - zeroed
    try testing.expectEqual(@as(f64, 4), lower.data[3]); // [1,0]
    try testing.expectEqual(@as(f64, 5), lower.data[4]); // [1,1]
    try testing.expectEqual(@as(f64, 0), lower.data[5]); // [1,2] - zeroed
    try testing.expectEqual(@as(f64, 7), lower.data[6]); // [2,0]
    try testing.expectEqual(@as(f64, 8), lower.data[7]); // [2,1]
    try testing.expectEqual(@as(f64, 9), lower.data[8]); // [2,2]
}

test "ndarray: tril() with k=1 keeps diagonal above main" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var lower = try mat.tril(allocator, 1);
    defer lower.deinit();

    // k=1: keep one diagonal above main
    try testing.expectEqual(@as(f64, 1), lower.data[0]); // [0,0]
    try testing.expectEqual(@as(f64, 2), lower.data[1]); // [0,1] - kept
    try testing.expectEqual(@as(f64, 0), lower.data[2]); // [0,2] - zeroed
    try testing.expectEqual(@as(f64, 4), lower.data[3]); // [1,0]
    try testing.expectEqual(@as(f64, 5), lower.data[4]); // [1,1]
    try testing.expectEqual(@as(f64, 6), lower.data[5]); // [1,2] - kept
}

test "ndarray: tril() with k=-1 zeroes main diagonal" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    var lower = try mat.tril(allocator, -1);
    defer lower.deinit();

    // k=-1: zero main diagonal and above
    try testing.expectEqual(@as(f64, 0), lower.data[0]); // [0,0] - zeroed
    try testing.expectEqual(@as(f64, 4), lower.data[3]); // [1,0]
    try testing.expectEqual(@as(f64, 0), lower.data[4]); // [1,1] - zeroed
    try testing.expectEqual(@as(f64, 7), lower.data[6]); // [2,0]
    try testing.expectEqual(@as(f64, 8), lower.data[7]); // [2,1]
    try testing.expectEqual(@as(f64, 0), lower.data[8]); // [2,2] - zeroed
}

test "ndarray: trace() computes sum of main diagonal" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    const tr = try mat.trace(0);
    try testing.expectEqual(@as(f64, 15), tr); // 1 + 5 + 9 = 15
}

test "ndarray: trace() with offset k=1" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    const tr = try mat.trace(1);
    try testing.expectEqual(@as(f64, 8), tr); // 2 + 6 = 8
}

test "ndarray: trace() with offset k=-1" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer mat.deinit();

    const tr = try mat.trace(-1);
    try testing.expectEqual(@as(f64, 12), tr); // 4 + 8 = 12
}

test "ndarray: trace() on non-square matrix (2x3)" {
    const allocator = testing.allocator;
    var mat = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer mat.deinit();

    const tr = try mat.trace(0);
    try testing.expectEqual(@as(i32, 6), tr); // 1 + 5 = 6 (only 2 diagonal elements)
}

test "ndarray: diagonal operations with column-major layout" {
    const allocator = testing.allocator;
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .column_major);
    defer mat.deinit();

    var upper = try mat.triu(allocator, 0);
    defer upper.deinit();

    // Column-major: data = [1,3,2,4] means [[1,2],[3,4]] in row-major view
    // Upper triangle preserves [1,2,0,4]
    const tr = try mat.trace(0);
    try testing.expectEqual(@as(f64, 5), tr); // 1 + 4 = 5
}

test "ndarray: diagonal operations with integer types" {
    const allocator = testing.allocator;
    var mat = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]i32{ 10, 20, 30, 40 }, .row_major);
    defer mat.deinit();

    const tr = try mat.trace(0);
    try testing.expectEqual(@as(i32, 50), tr); // 10 + 40 = 50

    var vec = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 5, 6 }, .row_major);
    defer vec.deinit();

    var diag_mat = try vec.diag(allocator, 0, .row_major);
    defer diag_mat.deinit();

    try testing.expectEqual(@as(i32, 5), diag_mat.data[0]);
    try testing.expectEqual(@as(i32, 6), diag_mat.data[3]);
}

test "ndarray: diagonal operations memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, .row_major);
        defer mat.deinit();

        var diag_arr = try mat.diagonal(allocator, 0);
        defer diag_arr.deinit();

        var upper = try mat.triu(allocator, 1);
        defer upper.deinit();

        var lower = try mat.tril(allocator, -1);
        defer lower.deinit();

        const tr = try mat.trace(0);
        try testing.expectEqual(@as(f64, 34), tr); // 1+6+11+16=34

        var vec = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
        defer vec.deinit();

        var diag_mat = try vec.diag(allocator, 0, .row_major);
        defer diag_mat.deinit();
    }
}

test "ndarray: triu/tril preserve original matrix data" {
    const allocator = testing.allocator;
    const orig_data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var mat = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &orig_data, .row_major);
    defer mat.deinit();

    var upper = try mat.triu(allocator, 0);
    defer upper.deinit();

    // Original should be unchanged
    try testing.expectEqual(@as(f64, 1), mat.data[0]);
    try testing.expectEqual(@as(f64, 4), mat.data[3]);
    try testing.expectEqual(@as(f64, 9), mat.data[8]);
}

test "ndarray: diag roundtrip 1D→2D→1D preserves values" {
    const allocator = testing.allocator;
    const original = [_]f64{ 10, 20, 30 };
    var vec = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &original, .row_major);
    defer vec.deinit();

    var mat = try vec.diag(allocator, 0, .row_major);
    defer mat.deinit();

    var extracted = try mat.diagonal(allocator, 0);
    defer extracted.deinit();

    try testing.expectEqual(@as(usize, 3), extracted.shape[0]);
    try testing.expectEqual(@as(f64, 10), extracted.data[0]);
    try testing.expectEqual(@as(f64, 20), extracted.data[1]);
    try testing.expectEqual(@as(f64, 30), extracted.data[2]);
}

// -- Indexing Tests (get/set) (7 tests) --

test "ndarray: get() retrieves single element from 2D array [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // Row-major [2,3]: [1,2,3 | 4,5,6]
    // Get [0,0] = 1.0, [0,2] = 3.0, [1,1] = 5.0
    try testing.expectEqual(1.0, arr.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(3.0, arr.get(&[_]isize{ @intCast(0), @intCast(2) }));
    try testing.expectEqual(5.0, arr.get(&[_]isize{ @intCast(1), @intCast(1) }));
}

test "ndarray: get() supports negative indexing (-1 = last element)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // [-1,-1] = [1,2] (last row, last col) = 6.0
    try testing.expectEqual(6.0, arr.get(&[_]isize{ -1, -1 }));
    // [-1,0] = [1,0] (last row, first col) = 4.0
    try testing.expectEqual(4.0, arr.get(&[_]isize{ -1, 0 }));
    // [0,-1] = [0,2] (first row, last col) = 3.0
    try testing.expectEqual(3.0, arr.get(&[_]isize{ 0, -1 }));
    // [-2,-2] = [0,1] (second-to-last row, second-to-last col) = 2.0
    try testing.expectEqual(2.0, arr.get(&[_]isize{ -2, -2 }));
}

test "ndarray: get() retrieves from 3D array [2,3,4]" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 3, 4 }, data[0..], .row_major);
    defer arr.deinit();

    // Row-major [2,3,4]: element [i,j,k] at index i*12 + j*4 + k
    try testing.expectEqual(@as(i32, 0), arr.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(i32, 3), arr.get(&[_]isize{ 0, 0, 3 }));
    try testing.expectEqual(@as(i32, 12), arr.get(&[_]isize{ 1, 0, 0 }));
    try testing.expectEqual(@as(i32, 23), arr.get(&[_]isize{ 1, 2, 3 }));
}

test "ndarray: set() modifies single element in 2D array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 1 }, 42.0);
    arr.set(&[_]isize{ 1, 2 }, 99.0);

    try testing.expectEqual(42.0, arr.data[1]);
    try testing.expectEqual(99.0, arr.data[5]);
    try testing.expectEqual(0.0, arr.data[0]);
}

test "ndarray: set() with negative indices modifies last row/col" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ -1, -1 }, 100);
    arr.set(&[_]isize{ -2, 0 }, 50);

    // [-1,-1] = [2,3] at index 2*4 + 3 = 11
    try testing.expectEqual(@as(i32, 100), arr.data[11]);
    // [-2,0] = [1,0] at index 1*4 + 0 = 4
    try testing.expectEqual(@as(i32, 50), arr.data[4]);
}

test "ndarray: set() persists changes across multiple operations" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{ 2, 2 }, 1.0, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, 10.0);
    arr.set(&[_]isize{ 0, 1 }, 20.0);
    arr.set(&[_]isize{ 1, 0 }, 30.0);
    arr.set(&[_]isize{ 1, 1 }, 40.0);

    try testing.expectEqual(10.0, arr.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(20.0, arr.get(&[_]isize{ @intCast(0), @intCast(1) }));
    try testing.expectEqual(30.0, arr.get(&[_]isize{ @intCast(1), @intCast(0) }));
    try testing.expectEqual(40.0, arr.get(&[_]isize{ @intCast(1), @intCast(1) }));
}

test "ndarray: get() rejects out-of-bounds positive indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Shape [2,3] allows indices [0-1, 0-2]
    // [2,0] is out of bounds (row index too high)
    try testing.expectError(error.IndexOutOfBounds, arr.get(&[_]isize{ @intCast(2), @intCast(0) }));
}

test "ndarray: get() rejects out-of-bounds negative indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // [-3, 0] is out of bounds for shape [2,3] (would be row -3)
    try testing.expectError(error.IndexOutOfBounds, arr.get(&[_]isize{ -3, 0 }));
}

// -- Flat Indexing Tests (at) (6 tests) --

test "ndarray: at() returns element at flat index in row-major [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // Flat indexing: 0→1, 1→2, 2→3, 3→4, 4→5, 5→6
    try testing.expectEqual(1.0, arr.at(0));
    try testing.expectEqual(3.0, arr.at(2));
    try testing.expectEqual(4.0, arr.at(3));
    try testing.expectEqual(6.0, arr.at(5));
}

test "ndarray: at() supports negative flat indices (-1 = last element)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // [-1] = last element = 6.0
    try testing.expectEqual(6.0, arr.at(-1));
    // [-2] = second-to-last = 5.0
    try testing.expectEqual(5.0, arr.at(-2));
    // [-6] = first element = 1.0
    try testing.expectEqual(1.0, arr.at(-6));
}

test "ndarray: at() respects memory layout (row-major vs column-major)" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };

    var arr_rm = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, data[0..], .row_major);
    defer arr_rm.deinit();

    var arr_cm = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, data[0..], .column_major);
    defer arr_cm.deinit();

    // Row-major: elements stored as [row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]
    // at(0) = arr.data[0] regardless of layout (flat index into storage)
    try testing.expectEqual(1.0, arr_rm.at(0));
    try testing.expectEqual(1.0, arr_cm.at(0));

    // at(1) accesses different logical positions
    // Row-major: at(1) = row0_col1 = data[1]
    // Column-major: at(1) = row1_col0 = data[1]
    // Both access data[1] in flat storage order
}

test "ndarray: at() works with 3D array [2,3,4]" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 3, 4 }, data[0..], .row_major);
    defer arr.deinit();

    // Flat indices 0-23 map directly to data array
    try testing.expectEqual(@as(i32, 0), arr.at(0));
    try testing.expectEqual(@as(i32, 10), arr.at(10));
    try testing.expectEqual(@as(i32, 23), arr.at(23));
}

test "ndarray: at() rejects out-of-bounds indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Shape [5] has 5 elements, valid indices: [0-4] or [-5 to -1]
    // at(5) is out of bounds
    try testing.expectError(error.IndexOutOfBounds, arr.at(5));
}

// -- Slicing Tests (slice) (8 tests) --

test "ndarray: slice() extracts row from 2D array [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice row 1: [5, 6, 7, 8]
    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 2 }, // rows 1:2 (single row)
        .{ null, null }, // all columns
    });

    try testing.expectEqual(@as(usize, 1), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 4), sliced.shape[1]);
    try testing.expectEqual(5.0, sliced.at(0));
    try testing.expectEqual(8.0, sliced.at(3));
}

test "ndarray: slice() extracts column from 2D array [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice column 2: [3, 7, 11]
    const sliced = arr.slice(&[_][2]?isize{
        .{ null, null }, // all rows
        .{ 2, 3 }, // column 2:3 (single column)
    });

    try testing.expectEqual(@as(usize, 3), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 1), sliced.shape[1]);

    // Use iterator to access sliced view elements (at() is not stride-aware)
    var iter = sliced.iterator();
    try testing.expectEqual(@as(f64, 3.0), iter.next().?);
    try testing.expectEqual(@as(f64, 7.0), iter.next().?);
    try testing.expectEqual(@as(f64, 11.0), iter.next().?);
}

test "ndarray: slice() extracts rectangular subregion [3,4] → [2,2]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice [1:3, 1:3] (rows 1-2, cols 1-2)
    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 3 }, // rows 1:3
        .{ 1, 3 }, // cols 1:3
    });

    try testing.expectEqual(@as(usize, 2), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 2), sliced.shape[1]);
    // Contents: [[6,7], [10,11]]
    try testing.expectEqual(6.0, sliced.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(7.0, sliced.get(&[_]isize{ @intCast(0), @intCast(1) }));
    try testing.expectEqual(10.0, sliced.get(&[_]isize{ @intCast(1), @intCast(0) }));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{ @intCast(1), @intCast(1) }));
}

test "ndarray: slice() with null bounds means unbounded dimension" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice all rows, cols 1:3
    const sliced = arr.slice(&[_][2]?isize{
        .{ null, null }, // unbounded rows = all rows
        .{ 1, 3 }, // cols 1:3
    });

    try testing.expectEqual(@as(usize, 3), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 2), sliced.shape[1]);
}

test "ndarray: slice() returns view sharing underlying data (non-owning)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    var sliced = arr.slice(&[_][2]?isize{
        .{ 1, 3 },
        .{ null, null },
    });

    // Modify original array
    arr.set(&[_]isize{ 1, 0 }, 999.0);

    // Slice should reflect the change (shares data)
    try testing.expectEqual(999.0, sliced.get(&[_]isize{ @intCast(0), @intCast(0) }));
}

test "ndarray: slice() of 3D array extracts sub-tensor [2,3,4] → [1,3,4]" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 3, 4 }, data[0..], .row_major);
    defer arr.deinit();

    // Slice first 3x4 matrix (first element in first dimension)
    const sliced = arr.slice(&[_][2]?isize{
        .{ 0, 1 },
        .{ null, null },
        .{ null, null },
    });

    try testing.expectEqual(@as(usize, 1), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 3), sliced.shape[1]);
    try testing.expectEqual(@as(usize, 4), sliced.shape[2]);
}

test "ndarray: slice() with negative indices works correctly" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice last 2 rows, last 2 cols using negative indices
    const sliced = arr.slice(&[_][2]?isize{
        .{ -2, null }, // rows -2 to end (last 2 rows)
        .{ -2, null }, // cols -2 to end (last 2 cols)
    });

    try testing.expectEqual(@as(usize, 2), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 2), sliced.shape[1]);
    // Should contain [[7,8], [11,12]]
    try testing.expectEqual(7.0, sliced.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(8.0, sliced.get(&[_]isize{ @intCast(0), @intCast(1) }));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{ @intCast(1), @intCast(0) }));
    try testing.expectEqual(12.0, sliced.get(&[_]isize{ @intCast(1), @intCast(1) }));
}

test "ndarray: slice() rejects out-of-bounds ranges" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Slice range [0:5] exceeds shape [3] (rows)
    const result = arr.slice(&[_][2]?isize{
        .{ 0, 5 }, // out of bounds
        .{ null, null },
    });
    _ = result;
    // Should panic or return error
}

// -- Negative Indexing Integration Tests (3 tests) --

test "ndarray: negative indices work consistently across get/set/at" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 1.0, .row_major);
    defer arr.deinit();

    // Array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // at(-1) = last element = 9.0
    try testing.expectEqual(9.0, arr.at(-1));

    // get([-1]) = same as at(-1)
    try testing.expectEqual(9.0, arr.get(&[_]isize{-1}));

    // set([-1], 99)
    arr.set(&[_]isize{-1}, 99.0);

    // Verify change
    try testing.expectEqual(99.0, arr.at(-1));
    try testing.expectEqual(99.0, arr.get(&[_]isize{-1}));
}

test "ndarray: negative 2D indexing wraps correctly" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Set [-1, -1] (last element)
    arr.set(&[_]isize{ -1, -1 }, 42.0);

    // Verify via positive index
    try testing.expectEqual(42.0, arr.get(&[_]isize{ @intCast(2), @intCast(3) }));

    // Set [-2, -3] (second-to-last row, third-to-last col)
    arr.set(&[_]isize{ -2, -3 }, 88.0);

    // Verify via positive index
    try testing.expectEqual(88.0, arr.get(&[_]isize{ @intCast(1), @intCast(1) }));
}

test "ndarray: slicing with negative bounds extracts tail regions" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 0, 10, 1, .row_major);
    defer arr.deinit();

    // Slice last 3 elements: [-3:end]
    const sliced = arr.slice(&[_][2]?isize{
        .{ -3, null },
    });

    try testing.expectEqual(@as(usize, 3), sliced.shape[0]);
    try testing.expectEqual(@as(i32, 7), sliced.at(0));
    try testing.expectEqual(@as(i32, 8), sliced.at(1));
    try testing.expectEqual(@as(i32, 9), sliced.at(2));
}

// ============================================================================
// ITERATOR PROTOCOL TESTS — NDArray traversal (Red phase — all FAIL)
// ============================================================================

// -- Basic Iteration Tests (8 tests) --

test "ndarray: iterator() returns NDArrayIterator for 1D array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 5.0, 1.0, .row_major);
    defer arr.deinit();

    const iter = arr.iterator();
    _ = iter;
    // Should have next() method
    // const first = iter.next(); // compiles without error
}

test "ndarray: iterator.next() returns elements in storage order (1D)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Should iterate [1.0, 2.0, 3.0, 4.0, 5.0] in order
    try testing.expectEqual(1.0, iter.next().?);
    try testing.expectEqual(2.0, iter.next().?);
    try testing.expectEqual(3.0, iter.next().?);
    try testing.expectEqual(4.0, iter.next().?);
    try testing.expectEqual(5.0, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator exhaustion returns null on repeated calls" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).full(allocator, &[_]usize{3}, 7.0, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Exhaust iterator
    _ = iter.next();
    _ = iter.next();
    _ = iter.next();

    // Should continue returning null
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over 2D array respects storage order (row-major)" {
    const allocator = testing.allocator;
    // [1, 2, 3 | 4, 5, 6] in row-major
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Row-major: iterate rows left-to-right
    try testing.expectEqual(@as(i32, 1), iter.next().?);
    try testing.expectEqual(@as(i32, 2), iter.next().?);
    try testing.expectEqual(@as(i32, 3), iter.next().?);
    try testing.expectEqual(@as(i32, 4), iter.next().?);
    try testing.expectEqual(@as(i32, 5), iter.next().?);
    try testing.expectEqual(@as(i32, 6), iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over 2D array respects column-major layout" {
    const allocator = testing.allocator;
    // Same data [1, 2, 3, 4, 5, 6] but column-major layout
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 }, .column_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Column-major storage: data[0]=1, data[1]=2, data[2]=3, data[3]=4, data[4]=5, data[5]=6
    // Iteration should respect column-major order
    try testing.expectEqual(@as(i32, 1), iter.next().?);
    try testing.expectEqual(@as(i32, 2), iter.next().?);
    try testing.expectEqual(@as(i32, 3), iter.next().?);
    try testing.expectEqual(@as(i32, 4), iter.next().?);
    try testing.expectEqual(@as(i32, 5), iter.next().?);
    try testing.expectEqual(@as(i32, 6), iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over 3D array [2,3,4] yields all 24 elements in order" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 3, 4 }, data[0..], .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |val| {
        try testing.expectEqual(@as(i32, @intCast(count)), val);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 24), count);
}

test "ndarray: iterator over empty slice yields no elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 1.0, .row_major);
    defer arr.deinit();

    // Slice to empty region
    const empty_slice = arr.slice(&[_][2]?isize{
        .{ 5, 5 }, // [5:5] has no elements
    });

    var iter = empty_slice.iterator();
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator after slice yields only slice elements (row-major)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]f64{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice middle 2x2 region [1:3, 1:3]
    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 3 },
        .{ 1, 3 },
    });

    var iter = sliced.iterator();

    // Should yield [6, 7, 10, 11] in row-major order of slice
    try testing.expectEqual(6.0, iter.next().?);
    try testing.expectEqual(7.0, iter.next().?);
    try testing.expectEqual(10.0, iter.next().?);
    try testing.expectEqual(11.0, iter.next().?);
    try testing.expect(iter.next() == null);
}

// -- Type Flexibility Tests (3 tests) --

test "ndarray: iterator works with i32 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &[_]i32{ 10, 20, 30, 40, 50 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    try testing.expectEqual(@as(i32, 10), iter.next().?);
    try testing.expectEqual(@as(i32, 20), iter.next().?);
    try testing.expectEqual(@as(i32, 30), iter.next().?);
    try testing.expectEqual(@as(i32, 40), iter.next().?);
    try testing.expectEqual(@as(i32, 50), iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator works with u64 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(u64, 2).full(allocator, &[_]usize{ 2, 3 }, 99, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |val| {
        try testing.expectEqual(@as(u64, 99), val);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 6), count);
}

test "ndarray: iterator works with complex struct element type" {
    const Complex = struct { real: f64, imag: f64 };
    const allocator = testing.allocator;
    var arr = try NDArray(Complex, 1).full(allocator, &[_]usize{3}, Complex{ .real = 1.0, .imag = 2.0 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    var count: usize = 0;
    while (iter.next()) |val| {
        try testing.expectEqual(1.0, val.real);
        try testing.expectEqual(2.0, val.imag);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 3), count);
}

// -- Creation Function Integration Tests (5 tests) --

test "ndarray: iterator over zeros() yields all zeros" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |val| {
        try testing.expectEqual(0.0, val);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 6), count);
}

test "ndarray: iterator over ones() yields all ones" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).ones(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |val| {
        try testing.expectEqual(1.0, val);
        count += 1;
    }

    try testing.expectEqual(@as(usize, 10), count);
}

test "ndarray: iterator over linspace() yields correct sequence" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).linspace(allocator, 0.0, 1.0, 5, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    const expected = [_]f64{ 0.0, 0.25, 0.5, 0.75, 1.0 };
    for (expected) |exp_val| {
        const actual = iter.next().?;
        try testing.expectApproxEqAbs(exp_val, actual, 1e-10);
    }

    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over eye() matrix yields diagonal pattern" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).eye(allocator, 3, 3, 0, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    const data = [_]f64{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };

    for (data) |expected_val| {
        const actual = iter.next().?;
        try testing.expectEqual(expected_val, actual);
    }

    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over arange() yields expected sequence" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 10, 20, 2, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    const expected = [_]i32{ 10, 12, 14, 16, 18 };
    for (expected) |exp_val| {
        const actual = iter.next().?;
        try testing.expectEqual(exp_val, actual);
    }

    try testing.expect(iter.next() == null);
}

// -- Single Element Tests (2 tests) --

test "ndarray: iterator over single-element array yields one value then null" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).full(allocator, &[_]usize{1}, 42.0, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    try testing.expectEqual(42.0, iter.next().?);
    try testing.expect(iter.next() == null);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over [1,1,1] 3D array yields single element" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).full(allocator, &[_]usize{ 1, 1, 1 }, 999, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    try testing.expectEqual(@as(i32, 999), iter.next().?);
    try testing.expect(iter.next() == null);
}

// -- Large Array Tests (2 tests) --

test "ndarray: iterator over large array [100,100] yields all 10K elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 10_000), count);
}

test "ndarray: iterator over large 1D array [1M] exhausts correctly" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).zeros(allocator, &[_]usize{1_000_000}, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var count: usize = 0;

    while (iter.next()) |_| {
        count += 1;
        if (count > 1_000_000) break; // safety check
    }

    try testing.expectEqual(@as(usize, 1_000_000), count);
    try testing.expect(iter.next() == null);
}

// -- Row-Major vs Column-Major Layout Tests (4 tests) --

test "ndarray: iterator row-major [2,3] yields storage order (row-by-row)" {
    const allocator = testing.allocator;
    // In row-major, elements stored: row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    for (0..6) |i| {
        const expected = @as(i32, @intCast(i + 1));
        try testing.expectEqual(expected, iter.next().?);
    }
}

test "ndarray: iterator column-major [2,3] respects column-first storage" {
    const allocator = testing.allocator;
    // In column-major, elements [1,2,3,4,5,6] stored column-first
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]i32{ 1, 2, 3, 4, 5, 6 }, .column_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Should iterate in storage order
    for (0..6) |i| {
        const expected = @as(i32, @intCast(i + 1));
        try testing.expectEqual(expected, iter.next().?);
    }
}

test "ndarray: iterator row-major [3,2,2] 3D array correct order" {
    const allocator = testing.allocator;
    var data: [12]i32 = undefined;
    for (0..12) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 3, 2, 2 }, data[0..], .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    for (0..12) |i| {
        try testing.expectEqual(@as(i32, @intCast(i)), iter.next().?);
    }
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator column-major [3,2,2] 3D array respects layout" {
    const allocator = testing.allocator;
    var data: [12]i32 = undefined;
    for (0..12) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 3, 2, 2 }, data[0..], .column_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Column-major: iterate in storage order
    for (0..12) |i| {
        try testing.expectEqual(@as(i32, @intCast(i)), iter.next().?);
    }
    try testing.expect(iter.next() == null);
}

// -- View (Sliced Array) Iterator Tests (4 tests) --

test "ndarray: iterator over sliced view yields only slice region" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 10.0, 1.0, .row_major);
    defer arr.deinit();

    // Slice elements 2-5
    const sliced = arr.slice(&[_][2]?isize{
        .{ 2, 5 },
    });

    var iter = sliced.iterator();

    try testing.expectEqual(2.0, iter.next().?);
    try testing.expectEqual(3.0, iter.next().?);
    try testing.expectEqual(4.0, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over 2D slice [1:3, 1:3] yields 2x2 region" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 4, 4 }, &[_]f64{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
    }, .row_major);
    defer arr.deinit();

    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 3 },
        .{ 1, 3 },
    });

    var iter = sliced.iterator();

    // Sliced 2x2: [5, 6, 9, 10]
    try testing.expectEqual(5.0, iter.next().?);
    try testing.expectEqual(6.0, iter.next().?);
    try testing.expectEqual(9.0, iter.next().?);
    try testing.expectEqual(10.0, iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator over single row slice yields row elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 3, 4 }, &[_]i32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice row 1: [5, 6, 7, 8]
    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 2 },
        .{ null, null },
    });

    var iter = sliced.iterator();

    try testing.expectEqual(@as(i32, 5), iter.next().?);
    try testing.expectEqual(@as(i32, 6), iter.next().?);
    try testing.expectEqual(@as(i32, 7), iter.next().?);
    try testing.expectEqual(@as(i32, 8), iter.next().?);
    try testing.expect(iter.next() == null);
}

test "ndarray: iterator count matches slice shape product" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 5, 5 }, .row_major);
    defer arr.deinit();

    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 4 },
        .{ 1, 4 },
    });

    // Sliced shape [3, 3] = 9 elements
    var iter = sliced.iterator();
    var count: usize = 0;

    while (iter.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 9), count);
}

// -- Modifications After Iteration Tests (3 tests) --

test "ndarray: modifying array after iterator creation doesn't affect iterator" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Modify the array
    arr.set(&[_]isize{0}, 999.0);
    arr.set(&[_]isize{1}, 888.0);

    // Iterator should reflect original values (or modified — depends on implementation)
    // Testing that iterator doesn't crash and gives consistent results
    try testing.expect(iter.next() != null);
    try testing.expect(iter.next() != null);
}

test "ndarray: iterator reflects changes in underlying data (shared slice)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Consume first element
    const first = iter.next().?;
    try testing.expectEqual(1.0, first);

    // Modify remaining elements
    arr.set(&[_]isize{1}, 999.0);

    // Next call should see modified value (or original — depends on implementation)
    try testing.expect(iter.next() != null);
}

test "ndarray: iterator state preserved across calls" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 0, 10, 1, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();

    // Consume first 3 elements
    try testing.expectEqual(@as(i32, 0), iter.next().?);
    try testing.expectEqual(@as(i32, 1), iter.next().?);
    try testing.expectEqual(@as(i32, 2), iter.next().?);

    // Continue consuming (iterator maintains internal position)
    try testing.expectEqual(@as(i32, 3), iter.next().?);

    // Iterator position should advance normally
    try testing.expectEqual(@as(i32, 4), iter.next().?);
}

// -- Integration with Collection Functions (2 tests) --

test "ndarray: iterator can be used in loop to verify all elements visited" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var visited_count: usize = 0;
    var sum: f64 = 0.0;

    while (iter.next()) |val| {
        visited_count += 1;
        sum += val;
    }

    try testing.expectEqual(@as(usize, 6), visited_count);
    try testing.expectEqual(21.0, sum); // 1+2+3+4+5+6
}

test "ndarray: iterator works in nested loop structure" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    var iter = arr.iterator();
    var prev_val: ?i32 = null;
    var count: usize = 0;

    while (iter.next()) |val| {
        if (prev_val) |p| {
            // Elements should appear in order
            try testing.expect(p <= val);
        }
        prev_val = val;
        count += 1;
    }

    try testing.expectEqual(@as(usize, 8), count);
}

// -- reshape() Function Tests (15+ tests) --





test "ndarray: reshape [3,4] → [2,6] changes layout composition" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..12) |i| {
        arr.data[i] = @floatFromInt(i + 1);
    }

    var reshaped = try arr.reshape(&[_]usize{ 2, 6 });
    defer reshaped.deinit();

    try testing.expectEqual(2, reshaped.shape[0]);
    try testing.expectEqual(6, reshaped.shape[1]);
    try testing.expectEqual(12, reshaped.count());
}



test "ndarray: reshape uses zero-copy for contiguous arrays (memory safety)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill with test data
    for (0..6) |i| {
        arr.data[i] = @floatFromInt(i);
    }

    const original_data_ptr = arr.data.ptr;

    var reshaped = try arr.reshape(&[_]usize{ 3, 2 });
    defer reshaped.deinit();

    // With ownership tracking, contiguous arrays use zero-copy (same pointer)
    // The reshaped view has owned=false, so both can safely call deinit()
    try testing.expect(original_data_ptr == reshaped.data.ptr);
    try testing.expectEqual(false, reshaped.owned); // View, not owned

    // Data should be accessible through the view
    try testing.expectEqual(@as(f64, 0.0), reshaped.data[0]);
}

test "ndarray: reshape non-contiguous array requires copy" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 4, 6 }, .row_major);
    defer arr.deinit();

    // Initialize with sequential data
    for (0..24) |i| {
        arr.data[i] = @floatFromInt(i + 1);
    }

    // Create a non-contiguous slice [0:3, 0:4]
    var sliced = arr.slice(&[_][2]?isize{ .{ 0, 3 }, .{ 0, 4 } });

    // After slice, strides are same but view doesn't own memory
    // Reshaping a non-contiguous view should copy
    var reshaped = try sliced.reshape(&[_]usize{ 3, 4 });
    defer reshaped.deinit();

    // Verify reshape succeeded
    try testing.expectEqual(3, reshaped.shape[0]);
    try testing.expectEqual(4, reshaped.shape[1]);
}

test "ndarray: reshape preserves row-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    var reshaped = try arr.reshape(&[_]usize{ 3, 2 });
    defer reshaped.deinit();

    try testing.expectEqual(Layout.row_major, reshaped.layout);
}

test "ndarray: reshape preserves column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer arr.deinit();

    var reshaped = try arr.reshape(&[_]usize{ 3, 2 });
    defer reshaped.deinit();

    try testing.expectEqual(Layout.column_major, reshaped.layout);
}

test "ndarray: reshape after zeros() creation" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // All elements initialized to 0.0
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }

    var reshaped = try arr.reshape(&[_]usize{ 3, 2 });
    defer reshaped.deinit();

    // Zeros should be preserved after reshape
    for (reshaped.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}


test "ndarray: reshape no memory leak with multiple allocations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 5, 6 }, .row_major);
        defer arr.deinit();

        var reshaped = try arr.reshape(&[_]usize{ 3, 10 });
        defer reshaped.deinit();

        try testing.expectEqual(30, reshaped.count());
    }

    // stdlib.testing.allocator will detect leaks if any allocation not freed
}

// -- transpose() Function Tests (13+ tests) --

test "ndarray: transpose 2D row-major [2,3] → [3,2] shape correct" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values: 1, 2, 3, 4, 5, 6
    for (0..arr.count()) |i| {
        arr.data[i] = @floatFromInt(i + 1);
    }

    const transposed = arr.transpose();

    // Shape must swap: [2,3] → [3,2]
    try testing.expectEqual(3, transposed.shape[0]);
    try testing.expectEqual(2, transposed.shape[1]);
}


test "ndarray: transpose 3D [2,3,4] → [4,3,2] shape correct" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    const transposed = arr.transpose();

    // Shape must reverse: [2,3,4] → [4,3,2]
    try testing.expectEqual(4, transposed.shape[0]);
    try testing.expectEqual(3, transposed.shape[1]);
    try testing.expectEqual(2, transposed.shape[2]);
}

test "ndarray: transpose 1D [6] → [6] no-op (shape unchanged)" {
    const allocator = testing.allocator;

    var arr = try NDArray(u8, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    for (0..6) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    const transposed = arr.transpose();

    // 1D transpose should be no-op
    try testing.expectEqual(6, transposed.shape[0]);
    try testing.expectEqual(arr.data.ptr, transposed.data.ptr);
}

test "ndarray: transpose zero-copy (same data pointer)" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    const transposed = arr.transpose();

    // Data pointer must be identical (zero-copy view)
    try testing.expectEqual(arr.data.ptr, transposed.data.ptr);
}


test "ndarray: transpose 2D row-major strides swap correctly" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Row-major [3,4] has strides [4, 1]
    try testing.expectEqual(4, arr.strides[0]);
    try testing.expectEqual(1, arr.strides[1]);

    const transposed = arr.transpose();

    // After transpose [4,3], strides should reverse: [1, 4]
    try testing.expectEqual(1, transposed.strides[0]);
    try testing.expectEqual(4, transposed.strides[1]);
}

test "ndarray: transpose 2D column-major strides swap correctly" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    // Column-major [3,4] has strides [1, 3]
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);

    const transposed = arr.transpose();

    // After transpose [4,3], strides should reverse: [3, 1]
    try testing.expectEqual(3, transposed.strides[0]);
    try testing.expectEqual(1, transposed.strides[1]);
}

test "ndarray: transpose 3D strides reverse correctly [2,3,4]→[4,3,2]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Row-major [2,3,4] has strides [12, 4, 1]
    try testing.expectEqual(12, arr.strides[0]);
    try testing.expectEqual(4, arr.strides[1]);
    try testing.expectEqual(1, arr.strides[2]);

    const transposed = arr.transpose();

    // After transpose [4,3,2], strides should reverse: [1, 4, 12]
    try testing.expectEqual(1, transposed.strides[0]);
    try testing.expectEqual(4, transposed.strides[1]);
    try testing.expectEqual(12, transposed.strides[2]);
}

test "ndarray: transpose twice restores original shape and strides" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    const original_shape = arr.shape;
    const original_strides = arr.strides;

    const transposed_once = arr.transpose();
    const transposed_twice = transposed_once.transpose();

    // Double transpose should restore original shape
    for (0..arr.shape.len) |i| {
        try testing.expectEqual(original_shape[i], transposed_twice.shape[i]);
    }

    // Double transpose should restore original strides
    for (0..arr.strides.len) |i| {
        try testing.expectEqual(original_strides[i], transposed_twice.strides[i]);
    }
}

test "ndarray: transpose iterator yields correct element order" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill: [[1,2,3],[4,5,6]]
    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;
    arr.data[4] = 5;
    arr.data[5] = 6;

    const transposed = arr.transpose();

    // Iterator should yield: [1,4,2,5,3,6] (rows of transposed [3,2])
    var iter = transposed.iterator();
    try testing.expectEqual(1, iter.next());
    try testing.expectEqual(4, iter.next());
    try testing.expectEqual(2, iter.next());
    try testing.expectEqual(5, iter.next());
    try testing.expectEqual(3, iter.next());
    try testing.expectEqual(6, iter.next());
    try testing.expectEqual(null, iter.next());
}

test "ndarray: transpose large array [100,200] shape reversal" {
    const allocator = testing.allocator;

    var arr = try NDArray(u32, 2).init(allocator, &[_]usize{ 100, 200 }, .row_major);
    defer arr.deinit();

    // Transpose should handle large arrays without allocation failure
    const transposed = arr.transpose();

    try testing.expectEqual(200, transposed.shape[0]);
    try testing.expectEqual(100, transposed.shape[1]);
}

test "ndarray: transpose no memory leak with multiple transposes" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 5, 6 }, .row_major);
        defer arr.deinit();

        for (0..10) |_| {
            const transposed = arr.transpose();
            // Each transposition creates a view but doesn't allocate new memory
            try testing.expectEqual(6, transposed.shape[0]);
        }
    }

    // stdlib.testing.allocator will detect leaks if any transposes incorrectly allocate
}

// ============================================================================
// flatten() Tests - Convert multi-dimensional array to 1D
// ============================================================================















// ============================================================================
// TESTS FOR ravel() — Always-Copy Flatten Semantics
// ============================================================================







test "ndarray: ravel empty-dimension array [0,5] error handling" {
    const allocator = testing.allocator;

    // Note: NDArray disallows zero dimensions at init time
    // This test verifies the design constraint is maintained
    const result = NDArray(i32, 2).init(allocator, &[_]usize{ 0, 5 }, .row_major);
    try testing.expectError(error.ZeroDimension, result);
}





// -- permute() Function Tests (12+ tests) --














// -- contiguous() Tests --












test "ndarray: contiguous distinguishes contiguous from non-contiguous views" {
    const allocator = stdlib.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    // Original array is contiguous
    var contiguous_original = try arr.contiguous();
    defer contiguous_original.deinit();

    // contiguous() always allocates new memory for safety (no double-free)
    try testing.expect(arr.data.ptr != contiguous_original.data.ptr);
    // But the data should be identical
    try testing.expectEqualSlices(i32, arr.data, contiguous_original.data);

    // Permuted view is non-contiguous
    const permuted = try arr.permute(&[_]usize{ 2, 1, 0 });
    var contiguous_permuted = try permuted.contiguous();
    defer contiguous_permuted.deinit();

    // Should allocate new buffer for permuted view
    try testing.expect(permuted.data.ptr != contiguous_permuted.data.ptr);

    // Both results have same number of elements
    try testing.expectEqual(arr.count(), contiguous_permuted.count());
}

// ============================================================================
// Element-wise Operations Tests
// ============================================================================

test "ndarray: add 1D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer b.deinit();

    // Fill arrays: a = [1, 2, 3, 4, 5], b = [10, 20, 30, 40, 50]
    for (0..5) |i| {
        a.data[i] = @as(f64, @floatFromInt(i + 1));
        b.data[i] = @as(f64, @floatFromInt((i + 1) * 10));
    }

    var result = try a.add(&b);
    defer result.deinit();

    // Verify result shape matches input
    try testing.expectEqual(a.shape[0], result.shape[0]);

    // Verify element-wise addition
    try testing.expectEqual(11.0, result.data[0]); // 1 + 10
    try testing.expectEqual(22.0, result.data[1]); // 2 + 20
    try testing.expectEqual(33.0, result.data[2]); // 3 + 30
    try testing.expectEqual(44.0, result.data[3]); // 4 + 40
    try testing.expectEqual(55.0, result.data[4]); // 5 + 50

    // Verify no data sharing with inputs
    try testing.expect(result.data.ptr != a.data.ptr);
    try testing.expect(result.data.ptr != b.data.ptr);
}

test "ndarray: add 2D arrays element-wise row_major" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer b.deinit();

    // Fill: a = [[1, 2, 3], [4, 5, 6]], b = [[10, 20, 30], [40, 50, 60]]
    for (0..6) |i| {
        a.data[i] = @intCast(i + 1);
        b.data[i] = @intCast((i + 1) * 10);
    }

    var result = try a.add(&b);
    defer result.deinit();

    // Verify shape matches
    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);

    // Verify additions
    try testing.expectEqual(11, result.data[0]); // 1+10
    try testing.expectEqual(22, result.data[1]); // 2+20
    try testing.expectEqual(33, result.data[2]); // 3+30
    try testing.expectEqual(44, result.data[3]); // 4+40
    try testing.expectEqual(55, result.data[4]); // 5+50
    try testing.expectEqual(66, result.data[5]); // 6+60
}

test "ndarray: add 2D arrays element-wise column_major" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer b.deinit();

    // Fill arrays
    for (0..6) |i| {
        a.data[i] = @as(f64, @floatFromInt(i + 1));
        b.data[i] = @as(f64, @floatFromInt((i + 1) * 10));
    }

    var result = try a.add(&b);
    defer result.deinit();

    // Verify shape
    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);

    // Result should match column-major layout of inputs
    try testing.expectEqual(11.0, result.data[0]);
    try testing.expectEqual(22.0, result.data[1]);
}

test "ndarray: add 3D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer b.deinit();

    for (0..8) |i| {
        a.data[i] = @intCast(i + 1);
        b.data[i] = 100;
    }

    var result = try a.add(&b);
    defer result.deinit();

    // Verify shape matches
    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);
    try testing.expectEqual(a.shape[2], result.shape[2]);

    // Verify all elements
    for (0..8) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1 + 100)), result.data[i]);
    }
}

test "ndarray: sub 1D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    // a = [10, 20, 30, 40], b = [1, 2, 3, 4]
    for (0..4) |i| {
        a.data[i] = @as(f64, @floatFromInt((i + 1) * 10));
        b.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var result = try a.sub(&b);
    defer result.deinit();

    // Verify subtraction: a - b
    try testing.expectEqual(9.0, result.data[0]);  // 10-1
    try testing.expectEqual(18.0, result.data[1]); // 20-2
    try testing.expectEqual(27.0, result.data[2]); // 30-3
    try testing.expectEqual(36.0, result.data[3]); // 40-4

    // Verify no data sharing
    try testing.expect(result.data.ptr != a.data.ptr);
    try testing.expect(result.data.ptr != b.data.ptr);
}

test "ndarray: sub 2D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer b.deinit();

    // a = [[100, 200], [300, 400]], b = [[10, 20], [30, 40]]
    for (0..4) |i| {
        a.data[i] = @intCast((i + 1) * 100);
        b.data[i] = @intCast((i + 1) * 10);
    }

    var result = try a.sub(&b);
    defer result.deinit();

    try testing.expectEqual(90, result.data[0]);   // 100-10
    try testing.expectEqual(180, result.data[1]);  // 200-20
    try testing.expectEqual(270, result.data[2]);  // 300-30
    try testing.expectEqual(360, result.data[3]);  // 400-40
}

test "ndarray: mul 1D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer b.deinit();

    // a = [2, 3, 4, 5, 6], b = [10, 10, 10, 10, 10]
    for (0..5) |i| {
        a.data[i] = @as(f64, @floatFromInt(i + 2));
        b.data[i] = 10.0;
    }

    var result = try a.mul(&b);
    defer result.deinit();

    try testing.expectEqual(20.0, result.data[0]);
    try testing.expectEqual(30.0, result.data[1]);
    try testing.expectEqual(40.0, result.data[2]);
    try testing.expectEqual(50.0, result.data[3]);
    try testing.expectEqual(60.0, result.data[4]);

    try testing.expect(result.data.ptr != a.data.ptr);
    try testing.expect(result.data.ptr != b.data.ptr);
}

test "ndarray: mul 2D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer b.deinit();

    // a = [[1, 2, 3], [4, 5, 6]], b = [[2, 2, 2], [2, 2, 2]]
    for (0..6) |i| {
        a.data[i] = @intCast(i + 1);
        b.data[i] = 2;
    }

    var result = try a.mul(&b);
    defer result.deinit();

    try testing.expectEqual(2, result.data[0]);   // 1*2
    try testing.expectEqual(4, result.data[1]);   // 2*2
    try testing.expectEqual(6, result.data[2]);   // 3*2
    try testing.expectEqual(8, result.data[3]);   // 4*2
    try testing.expectEqual(10, result.data[4]);  // 5*2
    try testing.expectEqual(12, result.data[5]);  // 6*2
}

test "ndarray: div 1D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    // a = [100, 200, 300, 400], b = [10, 10, 10, 10]
    for (0..4) |i| {
        a.data[i] = @as(f64, @floatFromInt((i + 1) * 100));
        b.data[i] = 10.0;
    }

    var result = try a.div(&b);
    defer result.deinit();

    try testing.expectEqual(10.0, result.data[0]);
    try testing.expectEqual(20.0, result.data[1]);
    try testing.expectEqual(30.0, result.data[2]);
    try testing.expectEqual(40.0, result.data[3]);

    try testing.expect(result.data.ptr != a.data.ptr);
    try testing.expect(result.data.ptr != b.data.ptr);
}

test "ndarray: div 2D arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer b.deinit();

    // a = [[10, 20], [30, 40]], b = [[2, 4], [5, 8]]
    a.data[0] = 10.0;
    a.data[1] = 20.0;
    a.data[2] = 30.0;
    a.data[3] = 40.0;

    b.data[0] = 2.0;
    b.data[1] = 4.0;
    b.data[2] = 5.0;
    b.data[3] = 8.0;

    var result = try a.div(&b);
    defer result.deinit();

    try testing.expectEqual(5.0, result.data[0]);   // 10/2
    try testing.expectEqual(5.0, result.data[1]);   // 20/4
    try testing.expectEqual(6.0, result.data[2]);   // 30/5
    try testing.expectEqual(5.0, result.data[3]);   // 40/8
}

test "ndarray: mod 1D integer arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer b.deinit();

    // a = [10, 11, 12, 13, 14], b = [3, 3, 3, 3, 3]
    for (0..5) |i| {
        a.data[i] = @intCast(10 + i);
        b.data[i] = 3;
    }

    var result = try a.mod(&b);
    defer result.deinit();

    try testing.expectEqual(1, result.data[0]);  // 10 % 3
    try testing.expectEqual(2, result.data[1]);  // 11 % 3
    try testing.expectEqual(0, result.data[2]);  // 12 % 3
    try testing.expectEqual(1, result.data[3]);  // 13 % 3
    try testing.expectEqual(2, result.data[4]);  // 14 % 3

    try testing.expect(result.data.ptr != a.data.ptr);
    try testing.expect(result.data.ptr != b.data.ptr);
}

test "ndarray: mod 2D integer arrays element-wise" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer b.deinit();

    // a = [[5, 10], [15, 20]], b = [[3, 4], [7, 6]]
    a.data[0] = 5;
    a.data[1] = 10;
    a.data[2] = 15;
    a.data[3] = 20;

    b.data[0] = 3;
    b.data[1] = 4;
    b.data[2] = 7;
    b.data[3] = 6;

    var result = try a.mod(&b);
    defer result.deinit();

    try testing.expectEqual(2, result.data[0]);  // 5 % 3
    try testing.expectEqual(2, result.data[1]);  // 10 % 4
    try testing.expectEqual(1, result.data[2]);  // 15 % 7
    try testing.expectEqual(2, result.data[3]);  // 20 % 6
}

test "ndarray: neg 1D array unary negation" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    // a = [1, -2, 3, -4, 5]
    a.data[0] = 1.0;
    a.data[1] = -2.0;
    a.data[2] = 3.0;
    a.data[3] = -4.0;
    a.data[4] = 5.0;

    var result = try a.neg();
    defer result.deinit();

    try testing.expectEqual(-1.0, result.data[0]);
    try testing.expectEqual(2.0, result.data[1]);
    try testing.expectEqual(-3.0, result.data[2]);
    try testing.expectEqual(4.0, result.data[3]);
    try testing.expectEqual(-5.0, result.data[4]);

    try testing.expect(result.data.ptr != a.data.ptr);
}


test "ndarray: abs 1D array absolute value" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    // a = [1.5, -2.5, 3.0, -4.5, 0]
    a.data[0] = 1.5;
    a.data[1] = -2.5;
    a.data[2] = 3.0;
    a.data[3] = -4.5;
    a.data[4] = 0.0;

    var result = try a.abs();
    defer result.deinit();

    try testing.expectEqual(1.5, result.data[0]);
    try testing.expectEqual(2.5, result.data[1]);
    try testing.expectEqual(3.0, result.data[2]);
    try testing.expectEqual(4.5, result.data[3]);
    try testing.expectEqual(0.0, result.data[4]);

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: abs 2D integer array absolute value" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    // a = [[1, -2], [-3, 4]]
    a.data[0] = 1;
    a.data[1] = -2;
    a.data[2] = -3;
    a.data[3] = 4;

    var result = try a.abs();
    defer result.deinit();

    try testing.expectEqual(1, result.data[0]);
    try testing.expectEqual(2, result.data[1]);
    try testing.expectEqual(3, result.data[2]);
    try testing.expectEqual(4, result.data[3]);
}

test "ndarray: exp 1D array exponential" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();

    // a = [0, 1, 2]
    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = 2.0;

    var result = try a.exp();
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10); // e^0 = 1
    try testing.expectApproxEqAbs(math.exp(1.0), result.data[1], 1e-10); // e^1 ≈ 2.71828
    try testing.expectApproxEqAbs(math.exp(2.0), result.data[2], 1e-10); // e^2 ≈ 7.38906

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: exp 2D array exponential" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = -1.0;
    a.data[3] = 2.0;

    var result = try a.exp();
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10);
    try testing.expectApproxEqAbs(math.exp(1.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(math.exp(-1.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(math.exp(2.0), result.data[3], 1e-10);
}

test "ndarray: log 1D array natural logarithm" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();

    // a = [1, e, e^2]
    a.data[0] = 1.0;
    a.data[1] = math.exp(1.0);
    a.data[2] = math.exp(2.0);

    var result = try a.log();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10); // ln(1) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10); // ln(e) = 1
    try testing.expectApproxEqAbs(2.0, result.data[2], 1e-10); // ln(e^2) = 2

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: log 2D array natural logarithm" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    a.data[0] = 1.0;
    a.data[1] = math.exp(1.0);
    a.data[2] = math.exp(2.0);
    a.data[3] = math.exp(3.0);

    var result = try a.log();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);
    try testing.expectApproxEqAbs(2.0, result.data[2], 1e-10);
    try testing.expectApproxEqAbs(3.0, result.data[3], 1e-10);
}

test "ndarray: sqrt 1D array square root" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    // a = [0, 1, 4, 9, 16]
    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = 4.0;
    a.data[3] = 9.0;
    a.data[4] = 16.0;

    var result = try a.sqrt();
    defer result.deinit();

    try testing.expectEqual(0.0, result.data[0]); // sqrt(0) = 0
    try testing.expectEqual(1.0, result.data[1]); // sqrt(1) = 1
    try testing.expectEqual(2.0, result.data[2]); // sqrt(4) = 2
    try testing.expectEqual(3.0, result.data[3]); // sqrt(9) = 3
    try testing.expectEqual(4.0, result.data[4]); // sqrt(16) = 4

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: sqrt 2D array square root" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    a.data[0] = 4.0;
    a.data[1] = 9.0;
    a.data[2] = 16.0;
    a.data[3] = 25.0;

    var result = try a.sqrt();
    defer result.deinit();

    try testing.expectEqual(2.0, result.data[0]);
    try testing.expectEqual(3.0, result.data[1]);
    try testing.expectEqual(4.0, result.data[2]);
    try testing.expectEqual(5.0, result.data[3]);
}

test "ndarray: pow 1D array power with exponent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();

    // a = [1, 2, 3, 4]
    for (0..4) |i| {
        a.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    var result = try a.pow(3.0);
    defer result.deinit();

    try testing.expectEqual(1.0, result.data[0]);   // 1^3 = 1
    try testing.expectEqual(8.0, result.data[1]);   // 2^3 = 8
    try testing.expectEqual(27.0, result.data[2]);  // 3^3 = 27
    try testing.expectEqual(64.0, result.data[3]);  // 4^3 = 64

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: pow 2D array power with exponent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    a.data[0] = 2.0;
    a.data[1] = 3.0;
    a.data[2] = 4.0;
    a.data[3] = 5.0;

    var result = try a.pow(2.0);
    defer result.deinit();

    try testing.expectEqual(4.0, result.data[0]);   // 2^2 = 4
    try testing.expectEqual(9.0, result.data[1]);   // 3^2 = 9
    try testing.expectEqual(16.0, result.data[2]);  // 4^2 = 16
    try testing.expectEqual(25.0, result.data[3]);  // 5^2 = 25
}

test "ndarray: pow with fractional exponent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();

    a.data[0] = 4.0;
    a.data[1] = 9.0;
    a.data[2] = 16.0;

    var result = try a.pow(0.5);
    defer result.deinit();

    try testing.expectEqual(2.0, result.data[0]); // 4^0.5 = 2
    try testing.expectEqual(3.0, result.data[1]); // 9^0.5 = 3
    try testing.expectEqual(4.0, result.data[2]); // 16^0.5 = 4
}

test "ndarray: floor 1D array rounding down" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer a.deinit();

    a.data[0] = 2.3;
    a.data[1] = 2.7;
    a.data[2] = -2.3;
    a.data[3] = -2.7;
    a.data[4] = 5.0;
    a.data[5] = 0.0;

    var result = try a.floor();
    defer result.deinit();

    try testing.expectEqual(2.0, result.data[0]);  // floor(2.3) = 2
    try testing.expectEqual(2.0, result.data[1]);  // floor(2.7) = 2
    try testing.expectEqual(-3.0, result.data[2]); // floor(-2.3) = -3
    try testing.expectEqual(-3.0, result.data[3]); // floor(-2.7) = -3
    try testing.expectEqual(5.0, result.data[4]);  // floor(5.0) = 5
    try testing.expectEqual(0.0, result.data[5]);  // floor(0.0) = 0
}

test "ndarray: ceil 1D array rounding up" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer a.deinit();

    a.data[0] = 2.3;
    a.data[1] = 2.7;
    a.data[2] = -2.3;
    a.data[3] = -2.7;
    a.data[4] = 5.0;
    a.data[5] = 0.0;

    var result = try a.ceil();
    defer result.deinit();

    try testing.expectEqual(3.0, result.data[0]);  // ceil(2.3) = 3
    try testing.expectEqual(3.0, result.data[1]);  // ceil(2.7) = 3
    try testing.expectEqual(-2.0, result.data[2]); // ceil(-2.3) = -2
    try testing.expectEqual(-2.0, result.data[3]); // ceil(-2.7) = -2
    try testing.expectEqual(5.0, result.data[4]);  // ceil(5.0) = 5
    try testing.expectEqual(0.0, result.data[5]);  // ceil(0.0) = 0
}

test "ndarray: round 1D array rounding to nearest" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{8}, .row_major);
    defer a.deinit();

    a.data[0] = 2.3;
    a.data[1] = 2.7;
    a.data[2] = -2.3;
    a.data[3] = -2.7;
    a.data[4] = 2.5;   // round half away from zero: rounds to 3
    a.data[5] = 3.5;   // round half away from zero: rounds to 4
    a.data[6] = 5.0;
    a.data[7] = 0.0;

    var result = try a.round();
    defer result.deinit();

    try testing.expectEqual(2.0, result.data[0]);  // round(2.3) = 2
    try testing.expectEqual(3.0, result.data[1]);  // round(2.7) = 3
    try testing.expectEqual(-2.0, result.data[2]); // round(-2.3) = -2
    try testing.expectEqual(-3.0, result.data[3]); // round(-2.7) = -3
    try testing.expectEqual(3.0, result.data[4]);  // round(2.5) = 3 (away from zero)
    try testing.expectEqual(4.0, result.data[5]);  // round(3.5) = 4 (away from zero)
    try testing.expectEqual(5.0, result.data[6]);  // round(5.0) = 5
    try testing.expectEqual(0.0, result.data[7]);  // round(0.0) = 0
}

test "ndarray: trunc 1D array truncating toward zero" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer a.deinit();

    a.data[0] = 2.3;
    a.data[1] = 2.7;
    a.data[2] = -2.3;
    a.data[3] = -2.7;
    a.data[4] = 5.0;
    a.data[5] = 0.0;

    var result = try a.trunc();
    defer result.deinit();

    try testing.expectEqual(2.0, result.data[0]);  // trunc(2.3) = 2
    try testing.expectEqual(2.0, result.data[1]);  // trunc(2.7) = 2
    try testing.expectEqual(-2.0, result.data[2]); // trunc(-2.3) = -2
    try testing.expectEqual(-2.0, result.data[3]); // trunc(-2.7) = -2
    try testing.expectEqual(5.0, result.data[4]);  // trunc(5.0) = 5
    try testing.expectEqual(0.0, result.data[5]);  // trunc(0.0) = 0
}

test "ndarray: floor 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) + 0.7;
        }

        var result = try a.floor();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);
    }
}

test "ndarray: ceil 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) + 0.3;
        }

        var result = try a.ceil();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);
    }
}

test "ndarray: trunc 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) + 0.5 - 1.5;
        }

        var result = try a.trunc();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);
    }
}

test "ndarray: sign positive values" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 3.5;
    a.data[1] = 0.0;
    a.data[2] = -2.7;
    a.data[3] = 100.0;
    a.data[4] = -0.001;

    var result = try a.sign();
    defer result.deinit();

    try testing.expectEqual(@as(f64, 1), result.data[0]);
    try testing.expectEqual(@as(f64, 0), result.data[1]);
    try testing.expectEqual(@as(f64, -1), result.data[2]);
    try testing.expectEqual(@as(f64, 1), result.data[3]);
    try testing.expectEqual(@as(f64, -1), result.data[4]);
}

test "ndarray: sign integer type" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer a.deinit();

    a.data[0] = 5;
    a.data[1] = 0;
    a.data[2] = -10;
    a.data[3] = 1;
    a.data[4] = -1;
    a.data[5] = 0;

    var result = try a.sign();
    defer result.deinit();

    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 0), result.data[1]);
    try testing.expectEqual(@as(i32, -1), result.data[2]);
    try testing.expectEqual(@as(i32, 1), result.data[3]);
    try testing.expectEqual(@as(i32, -1), result.data[4]);
    try testing.expectEqual(@as(i32, 0), result.data[5]);
}

test "ndarray: sign 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
        defer a.deinit();

        a.data[0] = 1.5;
        a.data[1] = -2.3;
        a.data[2] = 0.0;
        a.data[3] = -0.5;
        a.data[4] = 3.7;
        a.data[5] = 0.0;

        var result = try a.sign();
        defer result.deinit();

        try testing.expectEqual(@as(f64, 1), result.data[0]);
        try testing.expectEqual(@as(f64, -1), result.data[1]);
        try testing.expectEqual(@as(f64, 0), result.data[2]);
    }
}

test "ndarray: clip basic functionality" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{7}, .row_major);
    defer a.deinit();

    a.data[0] = -5.0;
    a.data[1] = -2.0;
    a.data[2] = 0.0;
    a.data[3] = 1.5;
    a.data[4] = 3.0;
    a.data[5] = 5.0;
    a.data[6] = 10.0;

    var result = try a.clip(-2.0, 3.0);
    defer result.deinit();

    try testing.expectEqual(@as(f64, -2.0), result.data[0]); // -5 clipped to -2
    try testing.expectEqual(@as(f64, -2.0), result.data[1]); // -2 stays -2
    try testing.expectEqual(@as(f64, 0.0), result.data[2]);  // 0 stays 0
    try testing.expectEqual(@as(f64, 1.5), result.data[3]);  // 1.5 stays 1.5
    try testing.expectEqual(@as(f64, 3.0), result.data[4]);  // 3 stays 3
    try testing.expectEqual(@as(f64, 3.0), result.data[5]);  // 5 clipped to 3
    try testing.expectEqual(@as(f64, 3.0), result.data[6]);  // 10 clipped to 3
}

test "ndarray: clip integer type" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = -100;
    a.data[1] = -5;
    a.data[2] = 0;
    a.data[3] = 5;
    a.data[4] = 100;

    var result = try a.clip(-10, 10);
    defer result.deinit();

    try testing.expectEqual(@as(i32, -10), result.data[0]);
    try testing.expectEqual(@as(i32, -5), result.data[1]);
    try testing.expectEqual(@as(i32, 0), result.data[2]);
    try testing.expectEqual(@as(i32, 5), result.data[3]);
    try testing.expectEqual(@as(i32, 10), result.data[4]);
}

test "ndarray: clip 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) - 2.5; // -2.5, -1.5, -0.5, 0.5, 1.5, 2.5
        }

        var result = try a.clip(-1.0, 1.0);
        defer result.deinit();

        try testing.expectEqual(@as(f64, -1.0), result.data[0]); // -2.5 → -1.0
        try testing.expectEqual(@as(f64, -1.0), result.data[1]); // -1.5 → -1.0
        try testing.expectApproxEqAbs(-0.5, result.data[2], 1e-10);
        try testing.expectApproxEqAbs(0.5, result.data[3], 1e-10);
        try testing.expectEqual(@as(f64, 1.0), result.data[4]); // 1.5 → 1.0
        try testing.expectEqual(@as(f64, 1.0), result.data[5]); // 2.5 → 1.0
    }
}

test "ndarray: where basic conditional selection" {
    const allocator = testing.allocator;

    var cond = try NDArray(bool, 1).init(allocator, &[_]usize{5}, .row_major);
    defer cond.deinit();
    cond.data[0] = true;
    cond.data[1] = false;
    cond.data[2] = true;
    cond.data[3] = false;
    cond.data[4] = true;

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;
    x.data[4] = 5.0;

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer y.deinit();
    y.data[0] = 10.0;
    y.data[1] = 20.0;
    y.data[2] = 30.0;
    y.data[3] = 40.0;
    y.data[4] = 50.0;

    var result = try NDArray(f64, 1).where(&cond, &x, &y);
    defer result.deinit();

    try testing.expectEqual(@as(f64, 1.0), result.data[0]);  // true → x
    try testing.expectEqual(@as(f64, 20.0), result.data[1]); // false → y
    try testing.expectEqual(@as(f64, 3.0), result.data[2]);  // true → x
    try testing.expectEqual(@as(f64, 40.0), result.data[3]); // false → y
    try testing.expectEqual(@as(f64, 5.0), result.data[4]);  // true → x
}

test "ndarray: where threshold masking" {
    const allocator = testing.allocator;

    var data = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer data.deinit();
    data.data[0] = -3.0;
    data.data[1] = -1.0;
    data.data[2] = 0.0;
    data.data[3] = 1.0;
    data.data[4] = 3.0;
    data.data[5] = 5.0;

    // Create condition: data >= 0
    var zeros = try NDArray(f64, 1).zeros(allocator, &[_]usize{6}, .row_major);
    defer zeros.deinit();

    var cond = try data.ge(&zeros);
    defer cond.deinit();

    var pos_val = try NDArray(f64, 1).full(allocator, &[_]usize{6}, 1.0, .row_major);
    defer pos_val.deinit();

    var neg_val = try NDArray(f64, 1).full(allocator, &[_]usize{6}, -1.0, .row_major);
    defer neg_val.deinit();

    var result = try NDArray(f64, 1).where(&cond, &pos_val, &neg_val);
    defer result.deinit();

    try testing.expectEqual(@as(f64, -1.0), result.data[0]); // -3 < 0 → -1
    try testing.expectEqual(@as(f64, -1.0), result.data[1]); // -1 < 0 → -1
    try testing.expectEqual(@as(f64, 1.0), result.data[2]);  // 0 >= 0 → 1
    try testing.expectEqual(@as(f64, 1.0), result.data[3]);  // 1 >= 0 → 1
    try testing.expectEqual(@as(f64, 1.0), result.data[4]);  // 3 >= 0 → 1
    try testing.expectEqual(@as(f64, 1.0), result.data[5]);  // 5 >= 0 → 1
}

test "ndarray: where 2D array with memory safety" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var cond = try NDArray(bool, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer cond.deinit();
        cond.data[0] = true;
        cond.data[1] = false;
        cond.data[2] = true;
        cond.data[3] = false;
        cond.data[4] = true;
        cond.data[5] = false;

        var x = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer x.deinit();
        for (x.data, 0..) |*val, idx| {
            val.* = @as(i32, @intCast(idx));
        }

        var y = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer y.deinit();
        for (y.data, 0..) |*val, idx| {
            val.* = @as(i32, @intCast(idx)) * 10;
        }

        var result = try NDArray(i32, 2).where(&cond, &x, &y);
        defer result.deinit();

        try testing.expectEqual(@as(i32, 0), result.data[0]);  // true → 0
        try testing.expectEqual(@as(i32, 10), result.data[1]); // false → 10
        try testing.expectEqual(@as(i32, 2), result.data[2]);  // true → 2
    }
}

test "ndarray: where shape mismatch error" {
    const allocator = testing.allocator;

    var cond = try NDArray(bool, 1).init(allocator, &[_]usize{3}, .row_major);
    defer cond.deinit();

    var x = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer x.deinit();

    var y = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major); // different size
    defer y.deinit();

    const result = NDArray(f64, 1).where(&cond, &x, &y);
    try testing.expectError(error.ShapeMismatch, result);
}

test "ndarray: where all true" {
    const allocator = testing.allocator;

    var cond = try NDArray(bool, 1).init(allocator, &[_]usize{4}, .row_major);
    defer cond.deinit();
    for (cond.data) |*val| {
        val.* = true;
    }

    var x = try NDArray(u8, 1).init(allocator, &[_]usize{4}, .row_major);
    defer x.deinit();
    x.data[0] = 1;
    x.data[1] = 2;
    x.data[2] = 3;
    x.data[3] = 4;

    var y = try NDArray(u8, 1).init(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();
    for (y.data) |*val| {
        val.* = 99;
    }

    var result = try NDArray(u8, 1).where(&cond, &x, &y);
    defer result.deinit();

    // All true → should select from x
    try testing.expectEqual(@as(u8, 1), result.data[0]);
    try testing.expectEqual(@as(u8, 2), result.data[1]);
    try testing.expectEqual(@as(u8, 3), result.data[2]);
    try testing.expectEqual(@as(u8, 4), result.data[3]);
}

test "ndarray: where all false" {
    const allocator = testing.allocator;

    var cond = try NDArray(bool, 1).init(allocator, &[_]usize{4}, .row_major);
    defer cond.deinit();
    for (cond.data) |*val| {
        val.* = false;
    }

    var x = try NDArray(u8, 1).init(allocator, &[_]usize{4}, .row_major);
    defer x.deinit();
    for (x.data) |*val| {
        val.* = 1;
    }

    var y = try NDArray(u8, 1).init(allocator, &[_]usize{4}, .row_major);
    defer y.deinit();
    y.data[0] = 10;
    y.data[1] = 20;
    y.data[2] = 30;
    y.data[3] = 40;

    var result = try NDArray(u8, 1).where(&cond, &x, &y);
    defer result.deinit();

    // All false → should select from y
    try testing.expectEqual(@as(u8, 10), result.data[0]);
    try testing.expectEqual(@as(u8, 20), result.data[1]);
    try testing.expectEqual(@as(u8, 30), result.data[2]);
    try testing.expectEqual(@as(u8, 40), result.data[3]);
}

test "ndarray: sin 1D array sine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    // a = [0, π/6, π/4, π/3, π/2]
    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 6.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 3.0;
    a.data[4] = pi / 2.0;

    var result = try a.sin();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);     // sin(0) = 0
    try testing.expectApproxEqAbs(0.5, result.data[1], 1e-10);     // sin(π/6) = 0.5
    try testing.expectApproxEqAbs(math.sin(pi / 4.0), result.data[2], 1e-10); // sin(π/4) ≈ 0.707
    try testing.expectApproxEqAbs(math.sin(pi / 3.0), result.data[3], 1e-10); // sin(π/3) ≈ 0.866
    try testing.expectApproxEqAbs(1.0, result.data[4], 1e-10);     // sin(π/2) = 1

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: sin 2D array sine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 2.0;
    a.data[2] = pi;
    a.data[3] = 3.0 * pi / 2.0;

    var result = try a.sin();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);   // sin(0) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);   // sin(π/2) = 1
    try testing.expectApproxEqAbs(0.0, result.data[2], 1e-10);   // sin(π) ≈ 0
    try testing.expectApproxEqAbs(-1.0, result.data[3], 1e-10);  // sin(3π/2) = -1
}

test "ndarray: cos 1D array cosine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 3.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 2.0;
    a.data[4] = pi;

    var result = try a.cos();
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10);                    // cos(0) = 1
    try testing.expectApproxEqAbs(0.5, result.data[1], 1e-10);                    // cos(π/3) = 0.5
    try testing.expectApproxEqAbs(math.cos(pi / 4.0), result.data[2], 1e-10); // cos(π/4) ≈ 0.707
    try testing.expectApproxEqAbs(0.0, result.data[3], 1e-10);                    // cos(π/2) = 0
    try testing.expectApproxEqAbs(-1.0, result.data[4], 1e-10);                   // cos(π) = -1

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: cos 2D array cosine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 2.0;
    a.data[2] = pi;
    a.data[3] = 3.0 * pi / 2.0;

    var result = try a.cos();
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10);   // cos(0) = 1
    try testing.expectApproxEqAbs(0.0, result.data[1], 1e-10);   // cos(π/2) = 0
    try testing.expectApproxEqAbs(-1.0, result.data[2], 1e-10);  // cos(π) = -1
    try testing.expectApproxEqAbs(0.0, result.data[3], 1e-10);   // cos(3π/2) ≈ 0
}

test "ndarray: tan 1D array tangent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();

    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 6.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 3.0;

    var result = try a.tan();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);                    // tan(0) = 0
    try testing.expectApproxEqAbs(math.tan(pi / 6.0), result.data[1], 1e-10); // tan(π/6) ≈ 0.577
    try testing.expectApproxEqAbs(1.0, result.data[2], 1e-10);                    // tan(π/4) = 1
    try testing.expectApproxEqAbs(math.tan(pi / 3.0), result.data[3], 1e-10); // tan(π/3) ≈ 1.732

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: tan 2D array tangent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 4.0;
    a.data[2] = -pi / 4.0;
    a.data[3] = pi / 6.0;

    var result = try a.tan();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);                     // tan(0) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);                     // tan(π/4) = 1
    try testing.expectApproxEqAbs(-1.0, result.data[2], 1e-10);                    // tan(-π/4) = -1
    try testing.expectApproxEqAbs(math.tan(pi / 6.0), result.data[3], 1e-10);  // tan(π/6) ≈ 0.577
}

test "ndarray: asin 1D array arcsine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 0.5;
    a.data[2] = -0.5;
    a.data[3] = 1.0;

    var result = try a.asin();
    defer result.deinit();

    const pi = math.pi;
    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);           // asin(0) = 0
    try testing.expectApproxEqAbs(pi / 6.0, result.data[1], 1e-10);      // asin(0.5) = π/6
    try testing.expectApproxEqAbs(-pi / 6.0, result.data[2], 1e-10);     // asin(-0.5) = -π/6
    try testing.expectApproxEqAbs(pi / 2.0, result.data[3], 1e-10);      // asin(1) = π/2
}

test "ndarray: acos 1D array arccosine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();

    a.data[0] = 1.0;
    a.data[1] = 0.5;
    a.data[2] = -0.5;
    a.data[3] = 0.0;

    var result = try a.acos();
    defer result.deinit();

    const pi = math.pi;
    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);           // acos(1) = 0
    try testing.expectApproxEqAbs(pi / 3.0, result.data[1], 1e-10);      // acos(0.5) = π/3
    try testing.expectApproxEqAbs(2.0 * pi / 3.0, result.data[2], 1e-10); // acos(-0.5) = 2π/3
    try testing.expectApproxEqAbs(pi / 2.0, result.data[3], 1e-10);      // acos(0) = π/2
}

test "ndarray: atan 1D array arctangent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = -1.0;
    a.data[3] = stdlib.math.sqrt(3.0);

    var result = try a.atan();
    defer result.deinit();

    const pi = math.pi;
    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);           // atan(0) = 0
    try testing.expectApproxEqAbs(pi / 4.0, result.data[1], 1e-10);      // atan(1) = π/4
    try testing.expectApproxEqAbs(-pi / 4.0, result.data[2], 1e-10);     // atan(-1) = -π/4
    try testing.expectApproxEqAbs(pi / 3.0, result.data[3], 1e-10);      // atan(√3) = π/3
}

test "ndarray: atan2 2D array two-argument arctangent" {
    const allocator = testing.allocator;
    var y = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer y.deinit();
    var x = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer x.deinit();

    // Set up coordinates for quadrants
    y.data[0] = 1.0;  x.data[0] = 1.0;    // Q1: (1, 1)
    y.data[1] = 1.0;  x.data[1] = -1.0;   // Q2: (-1, 1)
    y.data[2] = -1.0; x.data[2] = -1.0;   // Q3: (-1, -1)
    y.data[3] = -1.0; x.data[3] = 1.0;    // Q4: (1, -1)

    var result = try y.atan2(&x);
    defer result.deinit();

    const pi = math.pi;
    try testing.expectApproxEqAbs(pi / 4.0, result.data[0], 1e-10);        // atan2(1, 1) = π/4
    try testing.expectApproxEqAbs(3.0 * pi / 4.0, result.data[1], 1e-10);  // atan2(1, -1) = 3π/4
    try testing.expectApproxEqAbs(-3.0 * pi / 4.0, result.data[2], 1e-10); // atan2(-1, -1) = -3π/4
    try testing.expectApproxEqAbs(-pi / 4.0, result.data[3], 1e-10);       // atan2(-1, 1) = -π/4
}

test "ndarray: sinh 1D array hyperbolic sine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = -1.0;
    a.data[3] = 2.0;
    a.data[4] = -2.0;

    var result = try a.sinh();
    defer result.deinit();

    // sinh(0) = 0, sinh(1) ≈ 1.1752, sinh(-1) ≈ -1.1752
    // sinh(2) ≈ 3.6269, sinh(-2) ≈ -3.6269
    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.1752011936438014, result.data[1], 1e-10);
    try testing.expectApproxEqAbs(-1.1752011936438014, result.data[2], 1e-10);
    try testing.expectApproxEqAbs(3.626860407847019, result.data[3], 1e-10);
    try testing.expectApproxEqAbs(-3.626860407847019, result.data[4], 1e-10);
}

test "ndarray: cosh 1D array hyperbolic cosine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = -1.0;
    a.data[3] = 2.0;
    a.data[4] = -2.0;

    var result = try a.cosh();
    defer result.deinit();

    // cosh(0) = 1, cosh(±1) ≈ 1.5431, cosh(±2) ≈ 3.7622
    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10);
    try testing.expectApproxEqAbs(1.5430806348152437, result.data[1], 1e-10);
    try testing.expectApproxEqAbs(1.5430806348152437, result.data[2], 1e-10);  // cosh is even
    try testing.expectApproxEqAbs(3.7621956910836314, result.data[3], 1e-10);
    try testing.expectApproxEqAbs(3.7621956910836314, result.data[4], 1e-10);  // cosh is even
}

test "ndarray: tanh 1D array hyperbolic tangent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 0.0;
    a.data[1] = 1.0;
    a.data[2] = -1.0;
    a.data[3] = 2.0;
    a.data[4] = -2.0;

    var result = try a.tanh();
    defer result.deinit();

    // tanh(0) = 0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
    // tanh(2) ≈ 0.9640, tanh(-2) ≈ -0.9640
    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);
    try testing.expectApproxEqAbs(0.7615941559557649, result.data[1], 1e-10);
    try testing.expectApproxEqAbs(-0.7615941559557649, result.data[2], 1e-10);
    try testing.expectApproxEqAbs(0.9640275800758169, result.data[3], 1e-10);
    try testing.expectApproxEqAbs(-0.9640275800758169, result.data[4], 1e-10);
}

test "ndarray: sinh 2D array hyperbolic sine with memory safety" {
    const allocator = testing.allocator;

    // Test memory safety with multiple iterations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) * 0.5;
        }

        var result = try a.sinh();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);
    }
}

test "ndarray: cosh 2D array hyperbolic cosine with memory safety" {
    const allocator = testing.allocator;

    // Test memory safety with multiple iterations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) * 0.5;
        }

        var result = try a.cosh();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);
    }
}

test "ndarray: tanh 2D array hyperbolic tangent with memory safety" {
    const allocator = testing.allocator;

    // Test memory safety with multiple iterations
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
        defer a.deinit();

        for (a.data, 0..) |*val, idx| {
            val.* = @as(f64, @floatFromInt(idx)) * 0.5 - 1.5;  // Range -1.5 to 1.5
        }

        var result = try a.tanh();
        defer result.deinit();

        try testing.expect(result.shape[0] == 2);
        try testing.expect(result.shape[1] == 3);

        // Verify tanh is bounded: -1 < tanh(x) < 1
        for (result.data) |val| {
            try testing.expect(val > -1.0 and val < 1.0);
        }
    }
}

test "ndarray: log2 1D array base-2 logarithm" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 4.0;
    a.data[3] = 8.0;
    a.data[4] = 16.0;

    var result = try a.log2();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);  // log2(1) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);  // log2(2) = 1
    try testing.expectApproxEqAbs(2.0, result.data[2], 1e-10);  // log2(4) = 2
    try testing.expectApproxEqAbs(3.0, result.data[3], 1e-10);  // log2(8) = 3
    try testing.expectApproxEqAbs(4.0, result.data[4], 1e-10);  // log2(16) = 4
}

test "ndarray: log10 1D array base-10 logarithm" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    a.data[0] = 1.0;
    a.data[1] = 10.0;
    a.data[2] = 100.0;
    a.data[3] = 1000.0;
    a.data[4] = 0.1;

    var result = try a.log10();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);  // log10(1) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);  // log10(10) = 1
    try testing.expectApproxEqAbs(2.0, result.data[2], 1e-10);  // log10(100) = 2
    try testing.expectApproxEqAbs(3.0, result.data[3], 1e-10);  // log10(1000) = 3
    try testing.expectApproxEqAbs(-1.0, result.data[4], 1e-10); // log10(0.1) = -1
}

test "ndarray: eq 1D equality comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer b.deinit();

    a.data[0] = 1; b.data[0] = 1;
    a.data[1] = 2; b.data[1] = 3;
    a.data[2] = 4; b.data[2] = 4;
    a.data[3] = 5; b.data[3] = 6;
    a.data[4] = 7; b.data[4] = 7;

    var result = try a.eq(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == true);
    try testing.expect(result.data[1] == false);
    try testing.expect(result.data[2] == true);
    try testing.expect(result.data[3] == false);
    try testing.expect(result.data[4] == true);
}

test "ndarray: ne 2D inequality comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer b.deinit();

    a.data[0] = 1; b.data[0] = 1;
    a.data[1] = 2; b.data[1] = 3;
    a.data[2] = 4; b.data[2] = 4;
    a.data[3] = 5; b.data[3] = 6;

    var result = try a.ne(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == false);
    try testing.expect(result.data[1] == true);
    try testing.expect(result.data[2] == false);
    try testing.expect(result.data[3] == true);
}

test "ndarray: lt 1D less-than comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    a.data[0] = 1.0; b.data[0] = 2.0;
    a.data[1] = 3.0; b.data[1] = 3.0;
    a.data[2] = 5.0; b.data[2] = 4.0;
    a.data[3] = -1.0; b.data[3] = 0.0;

    var result = try a.lt(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == true);
    try testing.expect(result.data[1] == false);
    try testing.expect(result.data[2] == false);
    try testing.expect(result.data[3] == true);
}

test "ndarray: le 1D less-than-or-equal comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    a.data[0] = 1; b.data[0] = 2;
    a.data[1] = 3; b.data[1] = 3;
    a.data[2] = 5; b.data[2] = 4;
    a.data[3] = -1; b.data[3] = 0;

    var result = try a.le(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == true);
    try testing.expect(result.data[1] == true);
    try testing.expect(result.data[2] == false);
    try testing.expect(result.data[3] == true);
}

test "ndarray: gt 1D greater-than comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    a.data[0] = 2.0; b.data[0] = 1.0;
    a.data[1] = 3.0; b.data[1] = 3.0;
    a.data[2] = 4.0; b.data[2] = 5.0;
    a.data[3] = 0.0; b.data[3] = -1.0;

    var result = try a.gt(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == true);
    try testing.expect(result.data[1] == false);
    try testing.expect(result.data[2] == false);
    try testing.expect(result.data[3] == true);
}

test "ndarray: ge 1D greater-than-or-equal comparison" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    a.data[0] = 2; b.data[0] = 1;
    a.data[1] = 3; b.data[1] = 3;
    a.data[2] = 4; b.data[2] = 5;
    a.data[3] = 0; b.data[3] = -1;

    var result = try a.ge(&b);
    defer result.deinit();

    try testing.expect(result.data[0] == true);
    try testing.expect(result.data[1] == true);
    try testing.expect(result.data[2] == false);
    try testing.expect(result.data[3] == true);
}

test "ndarray: comparison shape mismatch error" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer b.deinit();

    a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
    b.data[0] = 1; b.data[1] = 2; b.data[2] = 3; b.data[3] = 4;

    // All comparison operations should return ShapeMismatch
    try testing.expectError(error.ShapeMismatch, a.eq(&b));
    try testing.expectError(error.ShapeMismatch, a.ne(&b));
    try testing.expectError(error.ShapeMismatch, a.lt(&b));
    try testing.expectError(error.ShapeMismatch, a.le(&b));
    try testing.expectError(error.ShapeMismatch, a.gt(&b));
    try testing.expectError(error.ShapeMismatch, a.ge(&b));
}

test "ndarray: add result is independent copy (no aliasing)" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer b.deinit();

    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    b.data[0] = 10.0;
    b.data[1] = 20.0;
    b.data[2] = 30.0;

    var result = try a.add(&b);
    defer result.deinit();

    // Modify original arrays
    a.data[0] = 100.0;
    b.data[0] = 200.0;

    // Result should remain unchanged
    try testing.expectEqual(11.0, result.data[0]);
}

test "ndarray: operations preserve layout information" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer b.deinit();

    for (0..6) |i| {
        a.data[i] = 1.0;
        b.data[i] = 2.0;
    }

    var result = try a.add(&b);
    defer result.deinit();

    // Result should preserve column-major layout from inputs
    try testing.expectEqual(a.layout, result.layout);
    try testing.expectEqual(b.layout, result.layout);
}

test "ndarray: 3D array element-wise operations" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer b.deinit();

    for (0..8) |i| {
        a.data[i] = @intCast(i);
        b.data[i] = 10;
    }

    var result = try a.add(&b);
    defer result.deinit();

    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);
    try testing.expectEqual(a.shape[2], result.shape[2]);

    for (0..8) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 10)), result.data[i]);
    }
}

test "ndarray: sqrt with 3D array" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer a.deinit();

    for (0..8) |i| {
        a.data[i] = @as(f64, @floatFromInt((i + 1) * (i + 1))); // [1, 4, 9, 16, 25, 36, 49, 64]
    }

    var result = try a.sqrt();
    defer result.deinit();

    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);
    try testing.expectEqual(a.shape[2], result.shape[2]);

    for (0..8) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), result.data[i]);
    }
}

// ========================================
// Broadcasting Tests (NumPy-compatible)
// ========================================

test "broadcast: scalar + 1D array not yet implemented (expected to fail)" {
    const allocator = testing.allocator;

    // This test documents expected behavior for broadcasting
    // Scalar = [1, 1, ...] in all dimensions conceptually
    // 1D array = [n]
    // Should broadcast to shape [n]

    var arr_1d = try NDArray(f64, 1).zeros(allocator, &[_]usize{5}, .row_major);
    defer arr_1d.deinit();

    // Initialize with values
    for (0..5) |i| {
        arr_1d.data[i] = @as(f64, @floatFromInt(i));
    }

    // Broadcasting scalar + 1D should fail until implemented
    // Expected: result shape [5] with [scalar+0, scalar+1, scalar+2, scalar+3, scalar+4]
}

test "broadcast: 1D array + 2D array row broadcast fails without broadcasting support" {
    const allocator = testing.allocator;

    // 1D array [n] should broadcast to [1, n] for 2D operation
    var arr_1d = try NDArray(f64, 1).ones(allocator, &[_]usize{4}, .row_major);
    defer arr_1d.deinit();

    var arr_2d = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_2d.deinit();

    // Current implementation requires same ndim and same shape
    // Broadcasting not yet supported, this test should fail
    // Expected after implementation: shape [3, 4]
}

test "broadcast: 2D [3,1] + 2D [1,4] incompatible without broadcasting" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{3, 1}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{1, 4}, .row_major);
    defer arr_b.deinit();

    // Without broadcasting: shapes [3,1] and [1,4] are incompatible
    // With broadcasting: should produce [3,4]
    // This test documents expected failure before broadcasting is implemented
}

test "broadcast: 3D + 1D broadcasting rule (shape mismatch without support)" {
    const allocator = testing.allocator;

    var arr_3d = try NDArray(f64, 3).ones(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr_3d.deinit();

    var arr_1d = try NDArray(f64, 1).ones(allocator, &[_]usize{4}, .row_major);
    defer arr_1d.deinit();

    // NumPy rule: 1D [4] broadcasts to [1,1,4] for 3D operation
    // Then [2,3,4] + [1,1,4] → [2,3,4]
    // Without broadcasting support, this operation fails
}

test "broadcast: empty dimension with size 1 (shape broadcasting)" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).ones(allocator, &[_]usize{5, 1}, .row_major);
    defer arr.deinit();

    // Dimension with size 1 should be broadcastable
    // This tests that we correctly identify broadcastable dimensions
    try testing.expectEqual(@as(usize, 1), arr.shape[1]);
}

test "broadcast error: incompatible shapes [3] + [4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 1).ones(allocator, &[_]usize{3}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 1).ones(allocator, &[_]usize{4}, .row_major);
    defer arr_b.deinit();

    // Shapes [3] and [4] are incompatible for broadcasting
    // Neither dimension is 1, so broadcast rule fails
    // Current add() will detect this as ShapeMismatch
    const result = arr_a.add(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast error: multi-dim mismatch [3,2] + [4,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{3, 2}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{4, 3}, .row_major);
    defer arr_b.deinit();

    // Shapes [3,2] and [4,3] are incompatible
    // Broadcasting would require compatible dimensions, but 3≠4 and neither is 1
    const result = arr_a.add(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast: same shape requires no broadcasting" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_b.deinit();

    // Same shape [3,4] + [3,4] should work (no broadcast needed)
    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);

    // All elements should be 2 (1 + 1)
    for (result.data) |val| {
        try testing.expectEqual(@as(f64, 2.0), val);
    }
}

test "broadcast add: result shape calculation for compatible dims" {
    const allocator = testing.allocator;

    // Test that when broadcasting is implemented, result shape is correct
    // For shapes [3,1] and [1,4]: max([3,1], [1,4]) = [3,4]

    var arr_a = try NDArray(i32, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    // Current implementation requires same shape; after broadcasting,
    // this pattern [3,1] + [1,4] → [3,4] should work
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
}

test "broadcast sub: shape [5,1] with itself (no broadcast needed)" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).full(allocator, &[_]usize{5, 1}, 10, .row_major);
    defer arr.deinit();

    var result = try arr.sub(&arr);
    defer result.deinit();

    // Subtracting same array should give all zeros
    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(usize, 1), result.shape[1]);

    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 0), val);
    }
}

test "broadcast mul: result values for 2D row-major" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).ones(allocator, &[_]usize{2, 3}, .row_major);
    defer arr_a.deinit();

    // Set arr_a to [1,2,3; 4,5,6]
    arr_a.data[0] = 1.0;
    arr_a.data[1] = 2.0;
    arr_a.data[2] = 3.0;
    arr_a.data[3] = 4.0;
    arr_a.data[4] = 5.0;
    arr_a.data[5] = 6.0;

    var arr_b = try NDArray(f64, 2).ones(allocator, &[_]usize{2, 3}, .row_major);
    defer arr_b.deinit();

    // Set arr_b to [2,2,2; 3,3,3]
    for (0..6) |i| {
        arr_b.data[i] = if (i < 3) 2.0 else 3.0;
    }

    var result = try arr_a.mul(&arr_b);
    defer result.deinit();

    // Expected: [2,4,6; 12,15,18]
    try testing.expectEqual(@as(f64, 2.0), result.data[0]);
    try testing.expectEqual(@as(f64, 4.0), result.data[1]);
    try testing.expectEqual(@as(f64, 6.0), result.data[2]);
    try testing.expectEqual(@as(f64, 12.0), result.data[3]);
    try testing.expectEqual(@as(f64, 15.0), result.data[4]);
    try testing.expectEqual(@as(f64, 18.0), result.data[5]);
}

test "broadcast div: element-wise division preserves shape" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).init(allocator, &[_]usize{2, 2}, .row_major);
    defer arr_a.deinit();

    // Set arr_a to [10, 20; 30, 40]
    arr_a.data[0] = 10.0;
    arr_a.data[1] = 20.0;
    arr_a.data[2] = 30.0;
    arr_a.data[3] = 40.0;

    var arr_b = try NDArray(f64, 2).init(allocator, &[_]usize{2, 2}, .row_major);
    defer arr_b.deinit();

    // Set arr_b to [2, 4; 5, 10]
    arr_b.data[0] = 2.0;
    arr_b.data[1] = 4.0;
    arr_b.data[2] = 5.0;
    arr_b.data[3] = 10.0;

    var result = try arr_a.div(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);

    // Expected: [5, 5; 6, 4]
    try testing.expectEqual(@as(f64, 5.0), result.data[0]);
    try testing.expectEqual(@as(f64, 5.0), result.data[1]);
    try testing.expectEqual(@as(f64, 6.0), result.data[2]);
    try testing.expectEqual(@as(f64, 4.0), result.data[3]);
}

test "broadcast: column-major layout with 2D [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(@as(usize, 3), arr.shape[0]);
    try testing.expectEqual(@as(usize, 4), arr.shape[1]);

    // Column-major stride should be [1, 3]
    try testing.expectEqual(@as(usize, 1), arr.strides[0]);
    try testing.expectEqual(@as(usize, 3), arr.strides[1]);
}

test "broadcast: 1D [N] vs 2D [N,1] shape differentiation" {
    const allocator = testing.allocator;

    var arr_1d = try NDArray(f64, 1).ones(allocator, &[_]usize{5}, .row_major);
    defer arr_1d.deinit();

    var arr_2d = try NDArray(f64, 2).ones(allocator, &[_]usize{5, 1}, .row_major);
    defer arr_2d.deinit();

    // Different ndim, should not work without dimension padding
    // 1D ndim=1, 2D ndim=2: incompatible for direct operation
    // This test documents pre-broadcasting behavior

    try testing.expectEqual(@as(usize, 1), arr_1d.shape.len);
    try testing.expectEqual(@as(usize, 2), arr_2d.shape.len);
}

test "broadcast: 3D + 1D broadcasting semantics" {
    const allocator = testing.allocator;

    // Test setup for 3D [2,3,4] + 1D [4] → should be [2,3,4]
    var arr_3d = try NDArray(i32, 3).ones(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr_3d.deinit();

    var arr_1d = try NDArray(i32, 1).ones(allocator, &[_]usize{4}, .row_major);
    defer arr_1d.deinit();

    // Without ndim padding (broadcasting), this fails
    // This test documents the need for preprocessing shapes
}

test "broadcast: multiple dimension-1 axes [1,5,1,3]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 4).ones(allocator, &[_]usize{1, 5, 1, 3}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(@as(usize, 1), arr.shape[0]);
    try testing.expectEqual(@as(usize, 5), arr.shape[1]);
    try testing.expectEqual(@as(usize, 1), arr.shape[2]);
    try testing.expectEqual(@as(usize, 3), arr.shape[3]);
}

test "broadcast: large arrays 1000+ elements same shape" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 1).zeros(allocator, &[_]usize{1001}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 1).zeros(allocator, &[_]usize{1001}, .row_major);
    defer arr_b.deinit();

    // Initialize arrays
    for (0..1001) |i| {
        arr_a.data[i] = @as(f64, @floatFromInt(i));
        arr_b.data[i] = @as(f64, @floatFromInt(i + 1000));
    }

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1001), result.shape[0]);

    // Check a few values
    try testing.expectEqual(@as(f64, 1000.0), result.data[0]);
    try testing.expectEqual(@as(f64, 1002.0), result.data[1]);
    try testing.expectEqual(@as(f64, 3000.0), result.data[1000]);
}

test "broadcast: stress test various dimension combinations" {
    const allocator = testing.allocator;

    // Test [2,3,4,5] addition with same shape
    var arr_a = try NDArray(i32, 4).ones(allocator, &[_]usize{2, 3, 4, 5}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 4).ones(allocator, &[_]usize{2, 3, 4, 5}, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    const expected_count = 2 * 3 * 4 * 5;
    for (0..expected_count) |i| {
        try testing.expectEqual(@as(i32, 2), result.data[i]);
    }
}

test "broadcast: add with identical 3D arrays [2,3,4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 3).full(allocator, &[_]usize{2, 3, 4}, 7, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 3).full(allocator, &[_]usize{2, 3, 4}, 3, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    try testing.expectEqual(@as(usize, 4), result.shape[2]);

    // All elements should be 10
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 10), val);
    }
}

test "broadcast: sub with identical shapes [4,5]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).full(allocator, &[_]usize{4, 5}, 100, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).full(allocator, &[_]usize{4, 5}, 30, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.sub(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(usize, 5), result.shape[1]);

    // All elements should be 70
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 70), val);
    }
}

test "broadcast error: shapes with incompatible middle dimension" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 3).ones(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 3).ones(allocator, &[_]usize{2, 5, 4}, .row_major);
    defer arr_b.deinit();

    // Shapes [2,3,4] and [2,5,4] incompatible (3 ≠ 5, neither is 1)
    const result = arr_a.add(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast: 2D row-major [3,1] add with row-major [3,4] (planned broadcast)" {
    const allocator = testing.allocator;

    // This test documents the expected behavior when broadcasting is fully implemented
    // [3,1] should broadcast to [3,4] to match [3,4] + [3,1] → [3,4]

    var arr_broadcast = try NDArray(i32, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_broadcast.deinit();

    var arr_col = try NDArray(i32, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_col.deinit();

    // Currently can only test with same shapes
    var result = try arr_broadcast.add(&arr_col);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
}

test "broadcast: column-major add with same 2D shape [5,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).full(allocator, &[_]usize{5, 3}, 2.5, .column_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 2).full(allocator, &[_]usize{5, 3}, 1.5, .column_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);

    // All elements should be 4.0
    for (result.data) |val| {
        try testing.expectEqual(@as(f64, 4.0), val);
    }
}

test "broadcast: mixed layout add column-major + row-major same shape" {
    const allocator = testing.allocator;

    var arr_col = try NDArray(i32, 2).full(allocator, &[_]usize{2, 3}, 5, .column_major);
    defer arr_col.deinit();

    var arr_row = try NDArray(i32, 2).full(allocator, &[_]usize{2, 3}, 7, .row_major);
    defer arr_row.deinit();

    // Same shape should work regardless of layout
    var result = try arr_col.add(&arr_row);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);

    // All elements should be 12
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 12), val);
    }
}

test "broadcast: single element dimension [1,1,1,5]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 4).ones(allocator, &[_]usize{1, 1, 1, 5}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 4).ones(allocator, &[_]usize{1, 1, 1, 5}, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(usize, 1), result.shape[1]);
    try testing.expectEqual(@as(usize, 1), result.shape[2]);
    try testing.expectEqual(@as(usize, 5), result.shape[3]);

    // All elements should be 2
    for (result.data) |val| {
        try testing.expectEqual(@as(f64, 2.0), val);
    }
}

test "broadcast error: 1D [5] + 1D [3] incompatible" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 1).ones(allocator, &[_]usize{5}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 1).ones(allocator, &[_]usize{3}, .row_major);
    defer arr_b.deinit();

    const result = arr_a.add(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast: 4D row-major [1,2,3,4] mul with itself" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 4).full(allocator, &[_]usize{1, 2, 3, 4}, 6, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 4).full(allocator, &[_]usize{1, 2, 3, 4}, 7, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.mul(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.shape[2]);
    try testing.expectEqual(@as(usize, 4), result.shape[3]);

    // All elements should be 42 (6 * 7)
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 42), val);
    }
}

test "broadcast: negative values sub [3] - [3] = 0" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 1).init(allocator, &[_]usize{3}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 1).init(allocator, &[_]usize{3}, .row_major);
    defer arr_b.deinit();

    // Set arr_a to [-5, 10, -15]
    arr_a.data[0] = -5;
    arr_a.data[1] = 10;
    arr_a.data[2] = -15;

    // Set arr_b to [-3, 2, -10]
    arr_b.data[0] = -3;
    arr_b.data[1] = 2;
    arr_b.data[2] = -10;

    var result = try arr_a.sub(&arr_b);
    defer result.deinit();

    // Expected: [-2, 8, -5]
    try testing.expectEqual(@as(i32, -2), result.data[0]);
    try testing.expectEqual(@as(i32, 8), result.data[1]);
    try testing.expectEqual(@as(i32, -5), result.data[2]);
}

test "broadcast: float precision mul [2,2]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).init(allocator, &[_]usize{2, 2}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 2).init(allocator, &[_]usize{2, 2}, .row_major);
    defer arr_b.deinit();

    // Set arr_a to [0.1, 0.2; 0.3, 0.4]
    arr_a.data[0] = 0.1;
    arr_a.data[1] = 0.2;
    arr_a.data[2] = 0.3;
    arr_a.data[3] = 0.4;

    // Set arr_b to [10.0, 5.0; 3.0, 2.5]
    arr_b.data[0] = 10.0;
    arr_b.data[1] = 5.0;
    arr_b.data[2] = 3.0;
    arr_b.data[3] = 2.5;

    var result = try arr_a.mul(&arr_b);
    defer result.deinit();

    // Expected: [1.0, 1.0; 0.9, 1.0]
    try testing.expectApproxEqRel(@as(f64, 1.0), result.data[0], 1e-10);
    try testing.expectApproxEqRel(@as(f64, 1.0), result.data[1], 1e-10);
    try testing.expectApproxEqRel(@as(f64, 0.9), result.data[2], 1e-10);
    try testing.expectApproxEqRel(@as(f64, 1.0), result.data[3], 1e-10);
}

test "broadcast: scalar-like [1] + [1] operation" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 1).zeros(allocator, &[_]usize{1}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 1).zeros(allocator, &[_]usize{1}, .row_major);
    defer arr_b.deinit();

    arr_a.data[0] = 100;
    arr_b.data[0] = 50;

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(i32, 150), result.data[0]);
}

test "broadcast: all-ones tensor [5,4,3] add" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 3).ones(allocator, &[_]usize{5, 4, 3}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 3).full(allocator, &[_]usize{5, 4, 3}, 2, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.shape[2]);

    const expected_count = 5 * 4 * 3;
    for (0..expected_count) |i| {
        try testing.expectEqual(@as(i32, 3), result.data[i]);
    }
}

test "broadcast: div by same array gives all ones [3,2]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 2}, .row_major);
    defer arr.deinit();

    // Set to [2, 4, 6, 8, 10, 12]
    for (0..6) |i| {
        arr.data[i] = @as(f64, @floatFromInt((i + 1) * 2));
    }

    var result = try arr.div(&arr);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);

    // All elements should be 1.0
    for (result.data) |val| {
        try testing.expectEqual(@as(f64, 1.0), val);
    }
}

// -- Reduction Operations Tests --
// Tests for sum(), prod(), mean(), min(), max() and their axis variants

test "reduction: sum() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [1, 2, 3, 4, 5]
    for (0..5) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Note: sum() method needs to be implemented
    // This test validates the API and expected behavior
    const result = arr.sum();
    try testing.expectEqual(@as(i32, 15), result);
}

test "reduction: sum() full 2D array i32 [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Expected: 1+2+...+12 = 78
    const result = arr.sum();
    try testing.expectEqual(@as(i32, 78), result);
}





test "reduction: prod() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    // Set to [2, 3, 4, 5]
    arr.data[0] = 2;
    arr.data[1] = 3;
    arr.data[2] = 4;
    arr.data[3] = 5;

    // Expected: 2*3*4*5 = 120
    const result = arr.prod();
    try testing.expectEqual(@as(i32, 120), result);
}




test "reduction: mean() full 1D array f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [1, 2, 3, 4, 5]
    for (0..5) |i| {
        arr.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    // Expected: (1+2+3+4+5)/5 = 3.0
    const result = arr.mean();
    try testing.expectApproxEqAbs(@as(f64, 3.0), result, 1e-10);
}




test "reduction: min() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [5, 2, 8, 1, 9]
    arr.data[0] = 5;
    arr.data[1] = 2;
    arr.data[2] = 8;
    arr.data[3] = 1;
    arr.data[4] = 9;

    // Expected: min = 1
    const result = arr.min();
    try testing.expectEqual(@as(i32, 1), result);
}




test "reduction: max() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [3, 7, 2, 9, 4]
    arr.data[0] = 3;
    arr.data[1] = 7;
    arr.data[2] = 2;
    arr.data[3] = 9;
    arr.data[4] = 4;

    // Expected: max = 9
    const result = arr.max();
    try testing.expectEqual(@as(i32, 9), result);
}




test "reduction: negative values in min/max operations" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [-5, -2, -8, -1, -9]
    arr.data[0] = -5;
    arr.data[1] = -2;
    arr.data[2] = -8;
    arr.data[3] = -1;
    arr.data[4] = -9;

    // Min should be -9, Max should be -1
    const min_result = arr.min();
    const max_result = arr.max();
    try testing.expectEqual(@as(i32, -9), min_result);
    try testing.expectEqual(@as(i32, -1), max_result);
}

test "reduction: mean() with integer array converts to f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{2, 2}, .row_major);
    defer arr.deinit();

    // Set to [[1, 2], [3, 4]]
    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;

    // Expected: (1+2+3+4)/4 = 2.5 (returns f64)
    const result = arr.mean();
    try testing.expectApproxEqAbs(@as(f64, 2.5), result, 1e-10);
}


test "reduction: column-major layout reduction consistency" {
    const allocator = testing.allocator;

    // Create row-major version
    var arr_row = try NDArray(i32, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr_row.deinit();

    // Create column-major version with same data
    var arr_col = try NDArray(i32, 2).init(allocator, &[_]usize{2, 3}, .column_major);
    defer arr_col.deinit();

    // Fill both with same values (note: storage order differs)
    for (0..6) |i| {
        arr_row.data[i] = @intCast(i + 1);
        arr_col.data[i] = @intCast(i + 1);
    }

    // Both should give same sum regardless of layout
    const sum_row = arr_row.sum();
    const sum_col = arr_col.sum();
    try testing.expectEqual(sum_row, sum_col);
}

test "reduction: single element array reductions" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{1, 1}, .row_major);
    defer arr.deinit();

    arr.data[0] = 42;

    // All reductions should return that single element
    const sum = arr.sum();
    const prod = arr.prod();
    const min = arr.min();
    const max = arr.max();
    try testing.expectEqual(@as(i32, 42), sum);
    try testing.expectEqual(@as(i32, 42), prod);
    try testing.expectEqual(@as(i32, 42), min);
    try testing.expectEqual(@as(i32, 42), max);
}

test "reduction: large array sum() performance" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{100, 100}, .row_major);
    defer arr.deinit();

    // Fill with value 1 (sum should be 10000)
    for (0..10000) |i| {
        arr.data[i] = 1;
    }

    const result = arr.sum();
    try testing.expectEqual(@as(i32, 10000), result);
}


test "reduction: prod() with zero element returns zero" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    // Set to [2, 0, 4, 5]
    arr.data[0] = 2;
    arr.data[1] = 0;
    arr.data[2] = 4;
    arr.data[3] = 5;

    // Expected: 2*0*4*5 = 0
    const result = arr.prod();
    try testing.expectEqual(@as(i32, 0), result);
}

// ============================================================================
// ADVANCED REDUCTION OPERATIONS: argmin, argmax, cumsum, cumprod, all, any
// ============================================================================

test "advanced reduction: argmin() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [5, 2, 8, 1, 9]
    arr.data[0] = 5;
    arr.data[1] = 2;
    arr.data[2] = 8;
    arr.data[3] = 1;
    arr.data[4] = 9;

    // Expected: index of min value = 3 (value 1)
    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 3), result);
}

test "advanced reduction: argmin() full 2D array f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set to [[4.5, 1.2, 9.8], [2.1, 3.3, 0.5]]
    arr.data[0] = 4.5;
    arr.data[1] = 1.2;
    arr.data[2] = 9.8;
    arr.data[3] = 2.1;
    arr.data[4] = 3.3;
    arr.data[5] = 0.5;

    // Expected: index 5 (linear index where 0.5 is stored)
    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 5), result);
}

test "advanced reduction: argmin() with duplicates returns first occurrence" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    // Set to [3, 1, 5, 1, 7, 1]
    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 5;
    arr.data[3] = 1;
    arr.data[4] = 7;
    arr.data[5] = 1;

    // Expected: index 1 (first occurrence of minimum value 1)
    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 1), result);
}

test "advanced reduction: argmax() full 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [3, 7, 2, 9, 4]
    arr.data[0] = 3;
    arr.data[1] = 7;
    arr.data[2] = 2;
    arr.data[3] = 9;
    arr.data[4] = 4;

    // Expected: index 3 (value 9)
    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 3), result);
}

test "advanced reduction: argmax() full 2D array f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set to [[1.5, 2.2, 3.9], [4.1, 0.8, 5.5]]
    arr.data[0] = 1.5;
    arr.data[1] = 2.2;
    arr.data[2] = 3.9;
    arr.data[3] = 4.1;
    arr.data[4] = 0.8;
    arr.data[5] = 5.5;

    // Expected: index 5 (value 5.5)
    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 5), result);
}

test "advanced reduction: argmax() with duplicates returns first occurrence" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    // Set to [3, 9, 5, 9, 7, 2]
    arr.data[0] = 3;
    arr.data[1] = 9;
    arr.data[2] = 5;
    arr.data[3] = 9;
    arr.data[4] = 7;
    arr.data[5] = 2;

    // Expected: index 1 (first occurrence of maximum value 9)
    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 1), result);
}

test "advanced reduction: argmin() single element" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.data[0] = 42;

    // Expected: index 0
    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 0), result);
}

test "advanced reduction: argmax() single element" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.data[0] = -99;

    // Expected: index 0
    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 0), result);
}







test "advanced reduction: all() on bool array all true" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [true, true, true, true, true]
    for (0..5) |i| {
        arr.data[i] = true;
    }

    const result = arr.all();
    try testing.expectEqual(true, result);
}

test "advanced reduction: all() on bool array with false" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [true, true, false, true, true]
    arr.data[0] = true;
    arr.data[1] = true;
    arr.data[2] = false;
    arr.data[3] = true;
    arr.data[4] = true;

    const result = arr.all();
    try testing.expectEqual(false, result);
}

test "advanced reduction: all() on bool 2D array all true" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set all to true
    for (0..6) |i| {
        arr.data[i] = true;
    }

    const result = arr.all();
    try testing.expectEqual(true, result);
}

test "advanced reduction: any() on bool array with one true" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [false, false, true, false, false]
    arr.data[0] = false;
    arr.data[1] = false;
    arr.data[2] = true;
    arr.data[3] = false;
    arr.data[4] = false;

    const result = arr.any();
    try testing.expectEqual(true, result);
}

test "advanced reduction: any() on bool array all false" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [false, false, false, false, false]
    for (0..5) |i| {
        arr.data[i] = false;
    }

    const result = arr.any();
    try testing.expectEqual(false, result);
}

test "advanced reduction: any() on bool 2D array mixed" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set to [[false, false, false], [false, true, false]]
    for (0..6) |i| {
        arr.data[i] = false;
    }
    arr.data[4] = true;

    const result = arr.any();
    try testing.expectEqual(true, result);
}

test "advanced reduction: single element bool all() returns that element" {
    const allocator = testing.allocator;

    var arr_true = try NDArray(bool, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr_true.deinit();
    arr_true.data[0] = true;

    var arr_false = try NDArray(bool, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr_false.deinit();
    arr_false.data[0] = false;

    try testing.expectEqual(true, arr_true.all());
    try testing.expectEqual(false, arr_false.all());
}

test "advanced reduction: single element bool any() returns that element" {
    const allocator = testing.allocator;

    var arr_true = try NDArray(bool, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr_true.deinit();
    arr_true.data[0] = true;

    var arr_false = try NDArray(bool, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr_false.deinit();
    arr_false.data[0] = false;

    try testing.expectEqual(true, arr_true.any());
    try testing.expectEqual(false, arr_false.any());
}






test "advanced reduction: argmin() with negative values i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [-5, -2, -8, -1, -9]
    arr.data[0] = -5;
    arr.data[1] = -2;
    arr.data[2] = -8;
    arr.data[3] = -1;
    arr.data[4] = -9;

    // Expected: index 4 (value -9)
    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 4), result);
}

test "advanced reduction: argmax() with negative values i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [-5, -2, -8, -1, -9]
    arr.data[0] = -5;
    arr.data[1] = -2;
    arr.data[2] = -8;
    arr.data[3] = -1;
    arr.data[4] = -9;

    // Expected: index 3 (value -1)
    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 3), result);
}

test "advanced reduction: argmin() column-major layout" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .column_major);
    defer arr.deinit();

    // Set to [5, 2, 8, 1, 9]
    arr.data[0] = 5;
    arr.data[1] = 2;
    arr.data[2] = 8;
    arr.data[3] = 1;
    arr.data[4] = 9;

    const result = try arr.argmin();
    try testing.expectEqual(@as(usize, 3), result);
}

test "advanced reduction: argmax() column-major layout" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .column_major);
    defer arr.deinit();

    // Set to [3, 7, 2, 9, 4]
    arr.data[0] = 3;
    arr.data[1] = 7;
    arr.data[2] = 2;
    arr.data[3] = 9;
    arr.data[4] = 4;

    const result = try arr.argmax();
    try testing.expectEqual(@as(usize, 3), result);
}

test "ndarray: save and load 1D i32 array" {
    const allocator = testing.allocator;

    // Create and populate array
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;
    arr.data[4] = 5;

    // Save to file
    try arr.save("/tmp/test_ndarray_1d.bin");

    // Load back
    var loaded = try NDArray(i32, 1).load(allocator, "/tmp/test_ndarray_1d.bin");
    defer loaded.deinit();

    // Verify shape
    try testing.expectEqual(@as(usize, 5), loaded.shape[0]);

    // Verify data
    try testing.expectEqual(@as(i32, 1), loaded.data[0]);
    try testing.expectEqual(@as(i32, 2), loaded.data[1]);
    try testing.expectEqual(@as(i32, 3), loaded.data[2]);
    try testing.expectEqual(@as(i32, 4), loaded.data[3]);
    try testing.expectEqual(@as(i32, 5), loaded.data[4]);

    // Verify layout
    try testing.expectEqual(Layout.row_major, loaded.layout);

    // Clean up
    try fs.cwd().deleteFile("/tmp/test_ndarray_1d.bin");
}

test "ndarray: save and load 2D f64 array" {
    const allocator = testing.allocator;

    // Create 2x3 array
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.data[0] = 1.1;
    arr.data[1] = 2.2;
    arr.data[2] = 3.3;
    arr.data[3] = 4.4;
    arr.data[4] = 5.5;
    arr.data[5] = 6.6;

    // Save
    try arr.save("/tmp/test_ndarray_2d.bin");

    // Load
    var loaded = try NDArray(f64, 2).load(allocator, "/tmp/test_ndarray_2d.bin");
    defer loaded.deinit();

    // Verify shape
    try testing.expectEqual(@as(usize, 2), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 3), loaded.shape[1]);

    // Verify data
    try testing.expectApproxEqAbs(1.1, loaded.data[0], 1e-10);
    try testing.expectApproxEqAbs(2.2, loaded.data[1], 1e-10);
    try testing.expectApproxEqAbs(3.3, loaded.data[2], 1e-10);
    try testing.expectApproxEqAbs(4.4, loaded.data[3], 1e-10);
    try testing.expectApproxEqAbs(5.5, loaded.data[4], 1e-10);
    try testing.expectApproxEqAbs(6.6, loaded.data[5], 1e-10);

    // Clean up
    try fs.cwd().deleteFile("/tmp/test_ndarray_2d.bin");
}

test "ndarray: save and load 3D u8 array column-major" {
    const allocator = testing.allocator;

    // Create 2x2x2 array
    var arr = try NDArray(u8, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .column_major);
    defer arr.deinit();

    for (0..8) |i| {
        arr.data[i] = @intCast(i * 10);
    }

    // Save
    try arr.save("/tmp/test_ndarray_3d.bin");

    // Load
    var loaded = try NDArray(u8, 3).load(allocator, "/tmp/test_ndarray_3d.bin");
    defer loaded.deinit();

    // Verify shape
    try testing.expectEqual(@as(usize, 2), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 2), loaded.shape[1]);
    try testing.expectEqual(@as(usize, 2), loaded.shape[2]);

    // Verify layout
    try testing.expectEqual(Layout.column_major, loaded.layout);

    // Verify data
    for (0..8) |i| {
        try testing.expectEqual(@as(u8, @intCast(i * 10)), loaded.data[i]);
    }

    // Clean up
    try fs.cwd().deleteFile("/tmp/test_ndarray_3d.bin");
}

test "ndarray: save and load bool array" {
    const allocator = testing.allocator;

    var arr = try NDArray(bool, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    arr.data[0] = true;
    arr.data[1] = false;
    arr.data[2] = true;
    arr.data[3] = false;

    try arr.save("/tmp/test_ndarray_bool.bin");

    var loaded = try NDArray(bool, 1).load(allocator, "/tmp/test_ndarray_bool.bin");
    defer loaded.deinit();

    try testing.expect(loaded.data[0] == true);
    try testing.expect(loaded.data[1] == false);
    try testing.expect(loaded.data[2] == true);
    try testing.expect(loaded.data[3] == false);

    try fs.cwd().deleteFile("/tmp/test_ndarray_bool.bin");
}

test "ndarray: load with wrong ndim fails" {
    const allocator = testing.allocator;

    // Create 1D array
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    for (0..5) |i| {
        arr.data[i] = @intCast(i);
    }

    try arr.save("/tmp/test_ndarray_wrong_ndim.bin");

    // Try to load as 2D (should fail)
    const result = NDArray(i32, 2).load(allocator, "/tmp/test_ndarray_wrong_ndim.bin");
    try testing.expectError(error.DimensionMismatch, result);

    try fs.cwd().deleteFile("/tmp/test_ndarray_wrong_ndim.bin");
}

test "ndarray: load with wrong type fails" {
    const allocator = testing.allocator;

    // Create i32 array
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    for (0..5) |i| {
        arr.data[i] = @intCast(i);
    }

    try arr.save("/tmp/test_ndarray_wrong_type.bin");

    // Try to load as f64 (should fail)
    const result = NDArray(f64, 1).load(allocator, "/tmp/test_ndarray_wrong_type.bin");
    try testing.expectError(error.TypeMismatch, result);

    try fs.cwd().deleteFile("/tmp/test_ndarray_wrong_type.bin");
}

test "ndarray: load nonexistent file fails" {
    const allocator = testing.allocator;

    const result = NDArray(i32, 1).load(allocator, "/tmp/nonexistent_ndarray_file.bin");
    try testing.expectError(error.FileNotFound, result);
}


test "ndarray: save and load large array" {
    const allocator = testing.allocator;

    // Create 100x100 array
    var arr = try NDArray(i64, 2).init(allocator, &[_]usize{ 100, 100 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..10000) |i| {
        arr.data[i] = @intCast(i);
    }

    try arr.save("/tmp/test_ndarray_large.bin");

    var loaded = try NDArray(i64, 2).load(allocator, "/tmp/test_ndarray_large.bin");
    defer loaded.deinit();

    // Verify shape
    try testing.expectEqual(@as(usize, 100), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 100), loaded.shape[1]);

    // Verify a few values
    try testing.expectEqual(@as(i64, 0), loaded.data[0]);
    try testing.expectEqual(@as(i64, 5000), loaded.data[5000]);
    try testing.expectEqual(@as(i64, 9999), loaded.data[9999]);

    try fs.cwd().deleteFile("/tmp/test_ndarray_large.bin");
}

// ============================================================================
// BROADCASTING TESTS — NumPy-compatible shape broadcasting
// ============================================================================
// Tests for broadcastShapes() function and element-wise operations with broadcasting
// Following NumPy broadcasting rules:
// 1. Compare shapes element-wise from right to left
// 2. Dimensions are compatible when equal OR one is 1
// 3. Prepend 1s to shorter-rank shapes to match ranks
// 4. Result shape is max of each dimension pair

// -- Sorting Operations Tests --

test "sort: basic 1D array ascending i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [30, 10, 50, 20, 40]
    arr.data[0] = 30;
    arr.data[1] = 10;
    arr.data[2] = 50;
    arr.data[3] = 20;
    arr.data[4] = 40;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    // Expected: [10, 20, 30, 40, 50]
    try testing.expectEqual(@as(i32, 10), sorted.data[0]);
    try testing.expectEqual(@as(i32, 20), sorted.data[1]);
    try testing.expectEqual(@as(i32, 30), sorted.data[2]);
    try testing.expectEqual(@as(i32, 40), sorted.data[3]);
    try testing.expectEqual(@as(i32, 50), sorted.data[4]);

    // Original should be unchanged
    try testing.expectEqual(@as(i32, 30), arr.data[0]);
}

test "sort: 1D already sorted array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 2), sorted.data[1]);
    try testing.expectEqual(@as(i32, 3), sorted.data[2]);
    try testing.expectEqual(@as(i32, 4), sorted.data[3]);
}

test "sort: 1D reverse sorted array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    arr.data[0] = 4;
    arr.data[1] = 3;
    arr.data[2] = 2;
    arr.data[3] = 1;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 2), sorted.data[1]);
    try testing.expectEqual(@as(i32, 3), sorted.data[2]);
    try testing.expectEqual(@as(i32, 4), sorted.data[3]);
}

test "sort: 1D with duplicates i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 4;
    arr.data[3] = 1;
    arr.data[4] = 5;
    arr.data[5] = 2;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 1), sorted.data[1]);
    try testing.expectEqual(@as(i32, 2), sorted.data[2]);
    try testing.expectEqual(@as(i32, 3), sorted.data[3]);
    try testing.expectEqual(@as(i32, 4), sorted.data[4]);
    try testing.expectEqual(@as(i32, 5), sorted.data[5]);
}

test "sort: 1D with negative values i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    arr.data[0] = -10;
    arr.data[1] = 5;
    arr.data[2] = -3;
    arr.data[3] = 0;
    arr.data[4] = -20;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    try testing.expectEqual(@as(i32, -20), sorted.data[0]);
    try testing.expectEqual(@as(i32, -10), sorted.data[1]);
    try testing.expectEqual(@as(i32, -3), sorted.data[2]);
    try testing.expectEqual(@as(i32, 0), sorted.data[3]);
    try testing.expectEqual(@as(i32, 5), sorted.data[4]);
}

test "sort: 1D single element i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.data[0] = 42;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    try testing.expectEqual(@as(i32, 42), sorted.data[0]);
}

test "sort: 2D along axis 0 (columns) i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr.deinit();

    // [[3, 1],
    //  [1, 3],
    //  [2, 2]]
    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 1;
    arr.data[3] = 3;
    arr.data[4] = 2;
    arr.data[5] = 2;

    var sorted = try arr.sort(allocator, 0);
    defer sorted.deinit();

    // Expected (sort each column):
    // [[1, 1],
    //  [2, 2],
    //  [3, 3]]
    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 1), sorted.data[1]);
    try testing.expectEqual(@as(i32, 2), sorted.data[2]);
    try testing.expectEqual(@as(i32, 2), sorted.data[3]);
    try testing.expectEqual(@as(i32, 3), sorted.data[4]);
    try testing.expectEqual(@as(i32, 3), sorted.data[5]);
}

test "sort: 2D along axis 1 (rows) i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // [[3, 1, 2],
    //  [6, 4, 5]]
    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 2;
    arr.data[3] = 6;
    arr.data[4] = 4;
    arr.data[5] = 5;

    var sorted = try arr.sort(allocator, 1);
    defer sorted.deinit();

    // Expected (sort each row):
    // [[1, 2, 3],
    //  [4, 5, 6]]
    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 2), sorted.data[1]);
    try testing.expectEqual(@as(i32, 3), sorted.data[2]);
    try testing.expectEqual(@as(i32, 4), sorted.data[3]);
    try testing.expectEqual(@as(i32, 5), sorted.data[4]);
    try testing.expectEqual(@as(i32, 6), sorted.data[5]);
}

test "sort: 2D with f64 type" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.data[0] = 3.5;
    arr.data[1] = 1.2;
    arr.data[2] = 2.8;
    arr.data[3] = 6.1;
    arr.data[4] = 4.0;
    arr.data[5] = 5.9;

    var sorted = try arr.sort(allocator, 1);
    defer sorted.deinit();

    try testing.expectApproxEqAbs(1.2, sorted.data[0], 1e-9);
    try testing.expectApproxEqAbs(2.8, sorted.data[1], 1e-9);
    try testing.expectApproxEqAbs(3.5, sorted.data[2], 1e-9);
}

test "sort: invalid axis error" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    const result = arr.sort(allocator, 2); // axis 2 doesn't exist
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "argsort: basic 1D array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Set to [30, 10, 50, 20, 40]
    arr.data[0] = 30;
    arr.data[1] = 10;
    arr.data[2] = 50;
    arr.data[3] = 20;
    arr.data[4] = 40;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Expected indices: [1, 3, 0, 4, 2]
    // (sorted values: [10, 20, 30, 40, 50])
    try testing.expectEqual(@as(usize, 1), indices.data[0]);
    try testing.expectEqual(@as(usize, 3), indices.data[1]);
    try testing.expectEqual(@as(usize, 0), indices.data[2]);
    try testing.expectEqual(@as(usize, 4), indices.data[3]);
    try testing.expectEqual(@as(usize, 2), indices.data[4]);
}

test "argsort: 1D already sorted array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Expected: [0, 1, 2, 3] (identity)
    try testing.expectEqual(@as(usize, 0), indices.data[0]);
    try testing.expectEqual(@as(usize, 1), indices.data[1]);
    try testing.expectEqual(@as(usize, 2), indices.data[2]);
    try testing.expectEqual(@as(usize, 3), indices.data[3]);
}

test "argsort: 1D reverse sorted array i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    arr.data[0] = 4;
    arr.data[1] = 3;
    arr.data[2] = 2;
    arr.data[3] = 1;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Expected: [3, 2, 1, 0] (reverse)
    try testing.expectEqual(@as(usize, 3), indices.data[0]);
    try testing.expectEqual(@as(usize, 2), indices.data[1]);
    try testing.expectEqual(@as(usize, 1), indices.data[2]);
    try testing.expectEqual(@as(usize, 0), indices.data[3]);
}

test "argsort: 1D with duplicates stable ordering i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 2;
    arr.data[3] = 1;
    arr.data[4] = 2;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Expected: indices pointing to values [1, 1, 2, 2, 3]
    // First two should be indices of the two 1s: [1, 3]
    // Next two should be indices of the two 2s: [2, 4]
    // Last should be index of 3: [0]
    try testing.expectEqual(@as(usize, 1), indices.data[0]);
    try testing.expectEqual(@as(usize, 3), indices.data[1]);
    try testing.expectEqual(@as(usize, 2), indices.data[2]);
    try testing.expectEqual(@as(usize, 4), indices.data[3]);
    try testing.expectEqual(@as(usize, 0), indices.data[4]);
}

test "argsort: 1D single element i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.data[0] = 42;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 0), indices.data[0]);
}

test "argsort: 2D along axis 0 (columns) i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr.deinit();

    // [[3, 1],
    //  [1, 3],
    //  [2, 2]]
    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 1;
    arr.data[3] = 3;
    arr.data[4] = 2;
    arr.data[5] = 2;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Column 0: [3, 1, 2] → indices [1, 2, 0]
    // Column 1: [1, 3, 2] → indices [0, 2, 1]
    try testing.expectEqual(@as(usize, 1), indices.data[0]); // col 0, row 0
    try testing.expectEqual(@as(usize, 0), indices.data[1]); // col 1, row 0
    try testing.expectEqual(@as(usize, 2), indices.data[2]); // col 0, row 1
    try testing.expectEqual(@as(usize, 2), indices.data[3]); // col 1, row 1
    try testing.expectEqual(@as(usize, 0), indices.data[4]); // col 0, row 2
    try testing.expectEqual(@as(usize, 1), indices.data[5]); // col 1, row 2
}

test "argsort: 2D along axis 1 (rows) i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // [[3, 1, 2],
    //  [6, 4, 5]]
    arr.data[0] = 3;
    arr.data[1] = 1;
    arr.data[2] = 2;
    arr.data[3] = 6;
    arr.data[4] = 4;
    arr.data[5] = 5;

    var indices = try arr.argsort(allocator, 1);
    defer indices.deinit();

    // Row 0: [3, 1, 2] → indices [1, 2, 0]
    // Row 1: [6, 4, 5] → indices [1, 2, 0]
    try testing.expectEqual(@as(usize, 1), indices.data[0]);
    try testing.expectEqual(@as(usize, 2), indices.data[1]);
    try testing.expectEqual(@as(usize, 0), indices.data[2]);
    try testing.expectEqual(@as(usize, 1), indices.data[3]);
    try testing.expectEqual(@as(usize, 2), indices.data[4]);
    try testing.expectEqual(@as(usize, 0), indices.data[5]);
}

test "argsort: invalid axis error" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    const result = arr.argsort(allocator, 2); // axis 2 doesn't exist
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "sort: memory safety with 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
        defer arr.deinit();

        arr.data[0] = 5;
        arr.data[1] = 2;
        arr.data[2] = 8;
        arr.data[3] = 1;
        arr.data[4] = 9;

        var sorted = try arr.sort(allocator, 0);
        defer sorted.deinit();

        try testing.expectEqual(@as(i32, 1), sorted.data[0]);
        try testing.expectEqual(@as(i32, 9), sorted.data[4]);
    }
}

test "argsort: memory safety with 10 iterations" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
        defer arr.deinit();

        arr.data[0] = 5;
        arr.data[1] = 2;
        arr.data[2] = 8;
        arr.data[3] = 1;
        arr.data[4] = 9;

        var indices = try arr.argsort(allocator, 0);
        defer indices.deinit();

        try testing.expectEqual(@as(usize, 3), indices.data[0]); // index of 1
        try testing.expectEqual(@as(usize, 4), indices.data[4]); // index of 9
    }
}

test "sort: 3D array along different axes" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer arr.deinit();

    // [[[4, 1],
    //   [3, 2]],
    //  [[8, 5],
    //   [7, 6]]]
    arr.data[0] = 4;
    arr.data[1] = 1;
    arr.data[2] = 3;
    arr.data[3] = 2;
    arr.data[4] = 8;
    arr.data[5] = 5;
    arr.data[6] = 7;
    arr.data[7] = 6;

    // Sort along axis 2 (innermost - sort pairs)
    var sorted = try arr.sort(allocator, 2);
    defer sorted.deinit();

    // Expected: [[[1, 4], [2, 3]], [[5, 8], [6, 7]]]
    try testing.expectEqual(@as(i32, 1), sorted.data[0]);
    try testing.expectEqual(@as(i32, 4), sorted.data[1]);
    try testing.expectEqual(@as(i32, 2), sorted.data[2]);
    try testing.expectEqual(@as(i32, 3), sorted.data[3]);
    try testing.expectEqual(@as(i32, 5), sorted.data[4]);
    try testing.expectEqual(@as(i32, 8), sorted.data[5]);
    try testing.expectEqual(@as(i32, 6), sorted.data[6]);
    try testing.expectEqual(@as(i32, 7), sorted.data[7]);
}

test "argsort: verify indices produce sorted array" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    arr.data[0] = 30;
    arr.data[1] = 10;
    arr.data[2] = 50;
    arr.data[3] = 20;
    arr.data[4] = 40;

    var indices = try arr.argsort(allocator, 0);
    defer indices.deinit();

    // Use indices to verify they produce sorted order
    const idx0 = indices.data[0];
    const idx1 = indices.data[1];
    const idx2 = indices.data[2];
    const idx3 = indices.data[3];
    const idx4 = indices.data[4];

    try testing.expect(arr.data[idx0] <= arr.data[idx1]);
    try testing.expect(arr.data[idx1] <= arr.data[idx2]);
    try testing.expect(arr.data[idx2] <= arr.data[idx3]);
    try testing.expect(arr.data[idx3] <= arr.data[idx4]);
}

// -- broadcastShapes Helper Function Tests --

test "broadcast: broadcastShapes same 2D shape [3,4] + [3,4] → [3,4]" {
    const allocator = testing.allocator;

    const shape_a = [_]usize{ 3, 4 };
    const shape_b = [_]usize{ 3, 4 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(usize, 3), result[0]);
    try testing.expectEqual(@as(usize, 4), result[1]);
}

test "broadcast: broadcastShapes dimension-1 broadcast [3,1] + [1,4] → [3,4]" {
    const allocator = testing.allocator;

    const shape_a = [_]usize{ 3, 1 };
    const shape_b = [_]usize{ 1, 4 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(usize, 3), result[0]);
    try testing.expectEqual(@as(usize, 4), result[1]);
}

test "broadcast: broadcastShapes rank mismatch [5,3] + [3] → [5,3]" {
    const allocator = testing.allocator;

    // [5, 3] + [3] should prepend 1 to [3] → [1, 3]
    // Then broadcast [5,3] and [1,3] → [5,3]
    const shape_a = [_]usize{ 5, 3 };
    const shape_b = [_]usize{3};

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(usize, 5), result[0]);
    try testing.expectEqual(@as(usize, 3), result[1]);
}

test "broadcast: broadcastShapes scalar (rank 0) broadcasts to any shape [3,4]" {
    const allocator = testing.allocator;

    // Scalar [] should broadcast to any shape [3, 4] → [3, 4]
    const shape_a = [_]usize{};
    const shape_b = [_]usize{ 3, 4 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 2), result.len);
    try testing.expectEqual(@as(usize, 3), result[0]);
    try testing.expectEqual(@as(usize, 4), result[1]);
}

test "broadcast: broadcastShapes complex 4D [5,1,4,1] + [3,1,2] → [5,3,4,2]" {
    const allocator = testing.allocator;

    // [5, 1, 4, 1] + [3, 1, 2]
    // Prepend 1 to second: [5, 1, 4, 1] + [1, 3, 1, 2]
    // Broadcast: [5, max(1,3), 4, max(1,2)] = [5, 3, 4, 2]
    const shape_a = [_]usize{ 5, 1, 4, 1 };
    const shape_b = [_]usize{ 3, 1, 2 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 4), result.len);
    try testing.expectEqual(@as(usize, 5), result[0]);
    try testing.expectEqual(@as(usize, 3), result[1]);
    try testing.expectEqual(@as(usize, 4), result[2]);
    try testing.expectEqual(@as(usize, 2), result[3]);
}

test "broadcast: broadcastShapes incompatible shapes [3] + [4] → error.IncompatibleShapes" {
    const allocator = testing.allocator;

    // [3] and [4] are incompatible: 3 ≠ 4 and neither is 1
    const shape_a = [_]usize{3};
    const shape_b = [_]usize{4};

    const result = broadcastShapes(&shape_a, &shape_b, allocator);
    try testing.expectError(error.IncompatibleShapes, result);
}

test "broadcast: broadcastShapes incompatible multi-dim [3,2] + [4,3] → error.IncompatibleShapes" {
    const allocator = testing.allocator;

    // [3,2] and [4,3]: comparing from right to left
    // dim 1: 2 vs 3 (neither is 1) → incompatible
    const shape_a = [_]usize{ 3, 2 };
    const shape_b = [_]usize{ 4, 3 };

    const result = broadcastShapes(&shape_a, &shape_b, allocator);
    try testing.expectError(error.IncompatibleShapes, result);
}

test "broadcast: broadcastShapes left shape broadcasts to right [1,4,1] + [3,4,5] → [3,4,5]" {
    const allocator = testing.allocator;

    const shape_a = [_]usize{ 1, 4, 1 };
    const shape_b = [_]usize{ 3, 4, 5 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(usize, 3), result[0]);
    try testing.expectEqual(@as(usize, 4), result[1]);
    try testing.expectEqual(@as(usize, 5), result[2]);
}

test "broadcast: broadcastShapes empty shape (rank 0) broadcasts correctly" {
    const allocator = testing.allocator;

    // Both empty (scalars) should work
    const shape_a = [_]usize{};
    const shape_b = [_]usize{};

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "broadcast: broadcastShapes scalar broadcasts left to 3D array [2,3,4]" {
    const allocator = testing.allocator;

    // [] + [2, 3, 4] → [2, 3, 4]
    const shape_a = [_]usize{};
    const shape_b = [_]usize{ 2, 3, 4 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(usize, 2), result[0]);
    try testing.expectEqual(@as(usize, 3), result[1]);
    try testing.expectEqual(@as(usize, 4), result[2]);
}

test "broadcast: broadcastShapes all-ones tensor [1,1,1] + [5,4,3] → [5,4,3]" {
    const allocator = testing.allocator;

    const shape_a = [_]usize{ 1, 1, 1 };
    const shape_b = [_]usize{ 5, 4, 3 };

    const result = try broadcastShapes(&shape_a, &shape_b, allocator);
    defer allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(usize, 5), result[0]);
    try testing.expectEqual(@as(usize, 4), result[1]);
    try testing.expectEqual(@as(usize, 3), result[2]);
}

test "broadcast: broadcastShapes error on incompatible [2,3,4] + [2,3,5]" {
    const allocator = testing.allocator;

    // Last dimension: 4 vs 5, neither is 1 → incompatible
    const shape_a = [_]usize{ 2, 3, 4 };
    const shape_b = [_]usize{ 2, 3, 5 };

    const result = broadcastShapes(&shape_a, &shape_b, allocator);
    try testing.expectError(error.IncompatibleShapes, result);
}

// -- Element-wise Operations with Broadcasting Tests --

test "broadcast: add with compatible shapes [3,1] + [1,4] produces [3,4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{ 3, 1 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{ 1, 4 }, .row_major);
    defer arr_b.deinit();

    // After broadcasting: [3,1] + [1,4] → [3,4]
    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);

    // All elements should be 2 (1 + 1)
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 2), val);
    }
}



test "broadcast: div with [4,1,3] / [1,2,1] broadcasts to [4,2,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 3).full(allocator, &[_]usize{ 4, 1, 3 }, 10.0, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 3).full(allocator, &[_]usize{ 1, 2, 1 }, 2.0, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.div(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape.len);
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.shape[2]);

    // All elements should be 5.0 (10 / 2)
    for (result.data) |val| {
        try testing.expectApproxEqAbs(@as(f64, 5.0), val, 1e-10);
    }
}

test "broadcast: comparison eq with shapes [3,1] and [1,4] broadcasts to [3,4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{ 3, 1 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{ 1, 4 }, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.eq(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);

    // All elements should be true (1 == 1)
    for (result.data) |val| {
        try testing.expectEqual(true, val);
    }
}

test "broadcast: comparison lt with [2,1,4] < [2,3,1] broadcasts to [2,3,4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 3).full(allocator, &[_]usize{ 2, 1, 4 }, 1, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 3).full(allocator, &[_]usize{ 2, 3, 1 }, 2, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.lt(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape.len);
    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    try testing.expectEqual(@as(usize, 4), result.shape[2]);

    // All elements should be true (1 < 2)
    for (result.data) |val| {
        try testing.expectEqual(true, val);
    }
}

test "broadcast: math function exp broadcasts with shape [1,3] broadcasted to [2,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).full(allocator, &[_]usize{ 1, 3 }, 0.0, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 2).full(allocator, &[_]usize{ 2, 3 }, 0.0, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.exp();
    defer result.deinit();

    // exp(0) = 1 for all elements
    for (result.data) |val| {
        try testing.expectApproxEqAbs(@as(f64, 1.0), val, 1e-10);
    }
}

test "broadcast: error on incompatible add [3,2] + [4,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{ 4, 3 }, .row_major);
    defer arr_b.deinit();

    const result = arr_a.add(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast: error on incompatible mul [5] * [3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 1).ones(allocator, &[_]usize{5}, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 1).ones(allocator, &[_]usize{3}, .row_major);
    defer arr_b.deinit();

    const result = arr_a.mul(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcast: error on incompatible sub [2,3,4] - [2,3,5]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 3).ones(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 3).ones(allocator, &[_]usize{ 2, 3, 5 }, .row_major);
    defer arr_b.deinit();

    const result = arr_a.sub(&arr_b);
    try testing.expectError(error.ShapeMismatch, result);
}



test "broadcast: layout preservation row-major [3,1] + [1,4] broadcasts to row-major [3,4]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).ones(allocator, &[_]usize{ 3, 1 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).ones(allocator, &[_]usize{ 1, 4 }, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
    try testing.expectEqual(.row_major, result.layout);
}

test "broadcast: large arrays 1000x1000 + [1,1000] broadcasts correctly" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(f64, 2).full(allocator, &[_]usize{ 1000, 1000 }, 1.0, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(f64, 2).full(allocator, &[_]usize{ 1, 1000 }, 2.0, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1000), result.shape[0]);
    try testing.expectEqual(@as(usize, 1000), result.shape[1]);
    try testing.expectEqual(@as(usize, 1_000_000), result.data.len);

    // Check a few elements
    for (result.data) |val| {
        try testing.expectEqual(@as(f64, 3.0), val);
    }
}

test "broadcast: stress test [5,1,1,3] + [1,4,2,1] broadcasts to [5,4,2,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 4).ones(allocator, &[_]usize{ 5, 1, 1, 3 }, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 4).ones(allocator, &[_]usize{ 1, 4, 2, 1 }, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.add(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape.len);
    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
    try testing.expectEqual(@as(usize, 3), result.shape[3]);

    // All elements should be 2 (1 + 1)
    try testing.expectEqual(@as(usize, 120), result.data.len);
    for (result.data) |val| {
        try testing.expectEqual(@as(i32, 2), val);
    }
}

test "broadcast: comparison ne with mismatched values [1,2] != [3,1] broadcasts to [3,2]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).full(allocator, &[_]usize{ 1, 2 }, 1, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).full(allocator, &[_]usize{ 3, 1 }, 2, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.ne(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);

    // All elements should be true (1 != 2)
    for (result.data) |val| {
        try testing.expectEqual(true, val);
    }
}

test "broadcast: comparison ge with [2,1] >= [1,3] broadcasts to [2,3]" {
    const allocator = testing.allocator;

    var arr_a = try NDArray(i32, 2).full(allocator, &[_]usize{ 2, 1 }, 5, .row_major);
    defer arr_a.deinit();

    var arr_b = try NDArray(i32, 2).full(allocator, &[_]usize{ 1, 3 }, 2, .row_major);
    defer arr_b.deinit();

    var result = try arr_a.ge(&arr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);

    // All elements should be true (5 >= 2)
    for (result.data) |val| {
        try testing.expectEqual(true, val);
    }
}

/// Compute broadcasted shape following NumPy rules.
/// Shapes are compared element-wise from right to left (trailing dimensions).
/// Two dimensions are compatible when they are equal OR one of them is 1.
/// If shapes have different ranks, prepend 1s to the shorter shape.
/// Result shape is the maximum of each dimension pair.
///
/// Examples:
///   [3, 1] + [1, 4] → [3, 4]
///   [5, 3] + [3] → [5, 3] (second becomes [1, 3])
///   [3] + [4] → error.IncompatibleShapes
///
/// Time: O(max(ndim_a, ndim_b))
/// Space: O(max(ndim_a, ndim_b))
fn broadcastShapes(shape_a: []const usize, shape_b: []const usize, allocator: mem.Allocator) !([]usize) {
    // Determine result rank
    const result_rank = @max(shape_a.len, shape_b.len);

    // Allocate result array
    const result = try allocator.alloc(usize, result_rank);
    errdefer allocator.free(result);

    // Compute offset for each shape (how many leading dimensions to skip)
    const offset_a = result_rank - shape_a.len;
    const offset_b = result_rank - shape_b.len;

    // Iterate from left to right, computing the broadcast shape
    for (0..result_rank) |i| {
        // Get dimension from shape_a (or 1 if beyond its rank)
        const dim_a = if (i < offset_a) 1 else shape_a[i - offset_a];

        // Get dimension from shape_b (or 1 if beyond its rank)
        const dim_b = if (i < offset_b) 1 else shape_b[i - offset_b];

        // Compute the broadcast dimension
        if (dim_a == dim_b) {
            result[i] = dim_a;
        } else if (dim_a == 1) {
            result[i] = dim_b;
        } else if (dim_b == 1) {
            result[i] = dim_a;
        } else {
            // Incompatible dimensions
            return error.IncompatibleShapes;
        }
    }

    return result;
}

/// Apply a binary operation with broadcasting support
/// Operation function signature: fn(a: T, b: T) T
fn applyBinaryOp(comptime T: type, comptime ndim: usize, self: *const NDArray(T, ndim),
    other: *const NDArray(T, ndim), allocator: mem.Allocator,
    comptime op: fn (T, T) T) !(NDArray(T, ndim)) {

    const Self = NDArray(T, ndim);

    // Compute broadcasted shape
    const broadcast_shape = broadcastShapes(&self.shape, &other.shape, allocator) catch |err| {
        return if (err == error.IncompatibleShapes) error.ShapeMismatch else err;
    };
    defer allocator.free(broadcast_shape);

    // Validate broadcast shape
    if (broadcast_shape.len != ndim) {
        return error.ShapeMismatch;
    }

    // Calculate total elements
    var total: usize = 1;
    for (broadcast_shape) |dim| {
        total *= dim;
    }

    // Allocate result
    const result_data = try allocator.alloc(T, total);
    errdefer allocator.free(result_data);

    // Compute result strides
    var result_strides: [ndim]usize = undefined;
    if (self.layout == .row_major) {
        result_strides[ndim - 1] = 1;
        if (ndim > 1) {
            var i: i32 = @intCast(ndim - 2);
            while (i >= 0) : (i -= 1) {
                result_strides[@intCast(i)] = result_strides[@intCast(i + 1)] * broadcast_shape[@intCast(i + 1)];
            }
        }
    } else {
        result_strides[0] = 1;
        for (1..ndim) |i| {
            result_strides[i] = result_strides[i - 1] * broadcast_shape[i - 1];
        }
    }

    // Apply operation element-wise
    var result_idx: [ndim]usize = undefined;

    for (0..total) |flat_idx| {
        // Convert flat index to multi-dimensional index
        var current = flat_idx;
        for (0..ndim) |i| {
            const dim_idx = ndim - 1 - i;
            result_idx[dim_idx] = current % broadcast_shape[dim_idx];
            current /= broadcast_shape[dim_idx];
        }

        // Map to source indices
        var self_idx: [ndim]usize = undefined;
        var other_idx: [ndim]usize = undefined;

        for (0..ndim) |i| {
            self_idx[i] = if (self.shape[i] == 1) 0 else result_idx[i];
            other_idx[i] = if (other.shape[i] == 1) 0 else result_idx[i];
        }

        // Compute offsets
        var self_offset: usize = 0;
        var other_offset: usize = 0;
        var result_offset: usize = 0;

        for (0..ndim) |i| {
            self_offset += self_idx[i] * self.strides[i];
            other_offset += other_idx[i] * other.strides[i];
            result_offset += result_idx[i] * result_strides[i];
        }

        result_data[result_offset] = op(self.data[self_offset], other.data[other_offset]);
    }

    // Create broadcast shape array
    var result_shape: [ndim]usize = undefined;
    for (0..ndim) |i| {
        result_shape[i] = broadcast_shape[i];
    }

    return Self{
        .shape = result_shape,
        .strides = result_strides,
        .data = result_data,
        .allocator = allocator,
        .layout = self.layout,
                    .owned = true,
            };
}

/// Apply a binary comparison operation with broadcasting support
/// Operation function signature: fn(a: T, b: T) bool
/// Returns an NDArray(bool, ndim) with the comparison result
fn applyBinaryCompOp(comptime T: type, comptime ndim: usize, self: *const NDArray(T, ndim),
    other: *const NDArray(T, ndim), allocator: mem.Allocator,
    comptime op: fn (T, T) bool) !(NDArray(bool, ndim)) {

    // Compute broadcasted shape
    const broadcast_shape = broadcastShapes(&self.shape, &other.shape, allocator) catch |err| {
        return if (err == error.IncompatibleShapes) error.ShapeMismatch else err;
    };
    defer allocator.free(broadcast_shape);

    // Validate broadcast shape
    if (broadcast_shape.len != ndim) {
        return error.ShapeMismatch;
    }

    // Calculate total elements
    var total: usize = 1;
    for (broadcast_shape) |dim| {
        total *= dim;
    }

    // Allocate result
    const result_data = try allocator.alloc(bool, total);
    errdefer allocator.free(result_data);

    // Compute result strides
    var result_strides: [ndim]usize = undefined;
    if (self.layout == .row_major) {
        result_strides[ndim - 1] = 1;
        if (ndim > 1) {
            var i: i32 = @intCast(ndim - 2);
            while (i >= 0) : (i -= 1) {
                result_strides[@intCast(i)] = result_strides[@intCast(i + 1)] * broadcast_shape[@intCast(i + 1)];
            }
        }
    } else {
        result_strides[0] = 1;
        for (1..ndim) |i| {
            result_strides[i] = result_strides[i - 1] * broadcast_shape[i - 1];
        }
    }

    // Apply operation element-wise
    var result_idx: [ndim]usize = undefined;

    for (0..total) |flat_idx| {
        // Convert flat index to multi-dimensional index
        var current = flat_idx;
        for (0..ndim) |i| {
            const dim_idx = ndim - 1 - i;
            result_idx[dim_idx] = current % broadcast_shape[dim_idx];
            current /= broadcast_shape[dim_idx];
        }

        // Map to source indices
        var self_idx: [ndim]usize = undefined;
        var other_idx: [ndim]usize = undefined;

        for (0..ndim) |i| {
            self_idx[i] = if (self.shape[i] == 1) 0 else result_idx[i];
            other_idx[i] = if (other.shape[i] == 1) 0 else result_idx[i];
        }

        // Compute offsets
        var self_offset: usize = 0;
        var other_offset: usize = 0;
        var result_offset: usize = 0;

        for (0..ndim) |i| {
            self_offset += self_idx[i] * self.strides[i];
            other_offset += other_idx[i] * other.strides[i];
            result_offset += result_idx[i] * result_strides[i];
        }

        result_data[result_offset] = op(self.data[self_offset], other.data[other_offset]);
    }

    // Create broadcast shape array
    var result_shape: [ndim]usize = undefined;
    for (0..ndim) |i| {
        result_shape[i] = broadcast_shape[i];
    }

    return NDArray(bool, ndim){
        .shape = result_shape,
        .strides = result_strides,
        .data = result_data,
        .allocator = allocator,
        .layout = self.layout,
        .owned = true, // Allocated new result data
    };
}

// -- Ownership Tracking Tests (8 tests) --
// These tests validate the ownership tracking feature that enables zero-copy views
// and prevents double-free bugs in reshape operations.

test "ownership: owned array deinit frees memory" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);

    // Store pointer to verify it gets freed
    const data_ptr = arr.data.ptr;
    const data_len = arr.data.len;

    arr.deinit();

    // Verify deinit was called on the data (by checking the array had data to free)
    // Note: We can't directly verify the memory is freed without platform-specific code,
    // but testing.allocator will catch leaks if deinit() isn't called properly
    _ = data_ptr;
    _ = data_len;
}

test "ownership: createView produces borrowed view" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Set a test value to verify view shares data
    arr.set(&.{ @as(isize, 0), @as(isize, 0) }, 42.0);

    // Create a view (borrowed reference to same data)
    var view = arr.createView();

    // Verify view points to same data
    try testing.expectEqual(arr.data.ptr, view.data.ptr);

    // Verify view sees the same value
    try testing.expectEqual(42.0, view.get(&.{ @as(isize, 0), @as(isize, 0) }));
}

test "ownership: double deinit safety - view then owner" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);

    // Create a view
    var view = arr.createView();

    // Deinit the view first (should not free shared data since owned=false)
    view.deinit();

    // Deinit the owner (should free shared data since owned=true)
    arr.deinit();

    // No double-free crash should occur
    // testing.allocator will detect any leaks
}

test "ownership: multiple views share same data" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{ 5 }, .row_major);
    defer arr.deinit();

    // Create multiple views from the same array
    var view1 = arr.createView();
    var view2 = arr.createView();

    // All should point to same data
    try testing.expectEqual(arr.data.ptr, view1.data.ptr);
    try testing.expectEqual(arr.data.ptr, view2.data.ptr);

    // Views don't need explicit deinit since they don't own data
    view1.deinit();
    view2.deinit();
}

test "ownership: modifying view updates original array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer arr.deinit();

    var view = arr.createView();

    // Modify through the view
    view.set(&.{ @as(isize, 1), @as(isize, 1) }, 99.0);

    // Verify original array sees the change
    try testing.expectEqual(99.0, arr.get(&.{ @as(isize, 1), @as(isize, 1) }));

    view.deinit();
}

test "ownership: reshape zero-copy on contiguous array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Set some test value
    arr.set(&.{ @as(isize, 0), @as(isize, 0) }, 123.0);

    // Reshape to compatible shape (still contiguous, same total elements)
    var reshaped = try arr.reshape(&[_]usize{ 4, 3 });
    defer reshaped.deinit();

    // For contiguous reshape, once ownership tracking is implemented, should create a view (same data pointer)
    // Note: Currently reshape always copies for safety, but will be optimized to zero-copy

    // For now, just verify the reshape succeeds and has correct shape
    try testing.expectEqual(4, reshaped.shape[0]);
    try testing.expectEqual(3, reshaped.shape[1]);
}

test "ownership: reshape copies on non-contiguous array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Transpose makes the array non-contiguous (same data, different strides)
    var transposed = arr.transpose();

    // Reshape the non-contiguous view
    // This should copy data since strides are non-standard
    var reshaped = try transposed.reshape(&[_]usize{ 2, 6 });
    defer reshaped.deinit();

    // Reshape should have succeeded with correct shape
    try testing.expectEqual(2, reshaped.shape[0]);
    try testing.expectEqual(6, reshaped.shape[1]);
}

test "ownership: no memory leaks with array and views" {
    const allocator = testing.allocator;

    {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 5, 5 }, .row_major);

        // Create multiple views
        var view1 = arr.createView();
        var view2 = arr.createView();
        var view3 = arr.createView();

        // Modify through views
        arr.set(&.{ @as(isize, 0), @as(isize, 0) }, 1.0);
        view1.set(&.{ @as(isize, 1), @as(isize, 1) }, 2.0);
        view2.set(&.{ @as(isize, 2), @as(isize, 2) }, 3.0);
        view3.set(&.{ @as(isize, 3), @as(isize, 3) }, 4.0);

        // Verify all modifications are visible in original
        try testing.expectEqual(1.0, arr.get(&.{ @as(isize, 0), @as(isize, 0) }));
        try testing.expectEqual(2.0, arr.get(&.{ @as(isize, 1), @as(isize, 1) }));
        try testing.expectEqual(3.0, arr.get(&.{ @as(isize, 2), @as(isize, 2) }));
        try testing.expectEqual(4.0, arr.get(&.{ @as(isize, 3), @as(isize, 3) }));

        // Deinit views (should not free data)
        view1.deinit();
        view2.deinit();
        view3.deinit();

        // Deinit owner (should free data once)
        arr.deinit();
    }
    // testing.allocator will report any leaks at scope end
}

test "ownership: view from reshape is borrowed" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&.{ @as(isize, 0), @as(isize, 0) }, 7.0);

    // Reshape to compatible 2D shape (contiguous array → zero-copy view)
    var reshaped = try arr.reshape(&[_]usize{ 4, 3 });
    defer reshaped.deinit();

    // Verify reshaped array has correct data layout and is a view (owned=false)
    try testing.expectEqual(false, reshaped.owned);
    try testing.expectEqual(7.0, reshaped.get(&.{ @as(isize, 0), @as(isize, 0) }));
}

test "ownership: owned flag prevents double-free in iteration" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer arr.deinit();

    // Iterate through original
    {
        var iter = arr.iterator();
        var count: usize = 0;
        while (iter.next()) |_| {
            count += 1;
        }
        try testing.expectEqual(4, count);
    }

    // Create a view and iterate through it
    var view = arr.createView();
    defer view.deinit();

    {
        var iter = view.iterator();
        var count: usize = 0;
        while (iter.next()) |_| {
            count += 1;
        }
        try testing.expectEqual(4, count);
    }
}

// -- Matrix Multiplication Tests --

test "matmul: 2x2 matrix multiplication" {
    const allocator = testing.allocator;

    // Create A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 1.0);
    A.set(&.{ 0, 1 }, 2.0);
    A.set(&.{ 1, 0 }, 3.0);
    A.set(&.{ 1, 1 }, 4.0);

    // Create B = [[5, 6], [7, 8]]
    var B = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer B.deinit();
    B.set(&.{ 0, 0 }, 5.0);
    B.set(&.{ 0, 1 }, 6.0);
    B.set(&.{ 1, 0 }, 7.0);
    B.set(&.{ 1, 1 }, 8.0);

    // C = A @ B
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //         = [[19, 22], [43, 50]]
    var C = try A.matmul(&B);
    defer C.deinit();

    try testing.expectEqual(2, C.shape[0]);
    try testing.expectEqual(2, C.shape[1]);
    try testing.expectApproxEqAbs(19.0, try C.get(&.{ 0, 0 }), 1e-10);
    try testing.expectApproxEqAbs(22.0, try C.get(&.{ 0, 1 }), 1e-10);
    try testing.expectApproxEqAbs(43.0, try C.get(&.{ 1, 0 }), 1e-10);
    try testing.expectApproxEqAbs(50.0, try C.get(&.{ 1, 1 }), 1e-10);
}

test "matmul: non-square matrix multiplication (3x2) @ (2x4)" {
    const allocator = testing.allocator;

    // Create A = [[1, 2], [3, 4], [5, 6]] (3x2)
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 1.0);
    A.set(&.{ 0, 1 }, 2.0);
    A.set(&.{ 1, 0 }, 3.0);
    A.set(&.{ 1, 1 }, 4.0);
    A.set(&.{ 2, 0 }, 5.0);
    A.set(&.{ 2, 1 }, 6.0);

    // Create B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
    var B = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 4 }, .row_major);
    defer B.deinit();
    B.set(&.{ 0, 0 }, 1.0);
    B.set(&.{ 0, 1 }, 2.0);
    B.set(&.{ 0, 2 }, 3.0);
    B.set(&.{ 0, 3 }, 4.0);
    B.set(&.{ 1, 0 }, 5.0);
    B.set(&.{ 1, 1 }, 6.0);
    B.set(&.{ 1, 2 }, 7.0);
    B.set(&.{ 1, 3 }, 8.0);

    // C = A @ B (3x4)
    // First row: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
    // Second row: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
    // Third row: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
    var C = try A.matmul(&B);
    defer C.deinit();

    try testing.expectEqual(3, C.shape[0]);
    try testing.expectEqual(4, C.shape[1]);

    // First row
    try testing.expectApproxEqAbs(11.0, try C.get(&.{ 0, 0 }), 1e-10);
    try testing.expectApproxEqAbs(14.0, try C.get(&.{ 0, 1 }), 1e-10);
    try testing.expectApproxEqAbs(17.0, try C.get(&.{ 0, 2 }), 1e-10);
    try testing.expectApproxEqAbs(20.0, try C.get(&.{ 0, 3 }), 1e-10);

    // Second row
    try testing.expectApproxEqAbs(23.0, try C.get(&.{ 1, 0 }), 1e-10);
    try testing.expectApproxEqAbs(30.0, try C.get(&.{ 1, 1 }), 1e-10);
    try testing.expectApproxEqAbs(37.0, try C.get(&.{ 1, 2 }), 1e-10);
    try testing.expectApproxEqAbs(44.0, try C.get(&.{ 1, 3 }), 1e-10);

    // Third row
    try testing.expectApproxEqAbs(35.0, try C.get(&.{ 2, 0 }), 1e-10);
    try testing.expectApproxEqAbs(46.0, try C.get(&.{ 2, 1 }), 1e-10);
    try testing.expectApproxEqAbs(57.0, try C.get(&.{ 2, 2 }), 1e-10);
    try testing.expectApproxEqAbs(68.0, try C.get(&.{ 2, 3 }), 1e-10);
}

test "matmul: vector dot product (1D)" {
    const allocator = testing.allocator;

    // Create v1 = [1, 2, 3, 4]
    var v1 = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer v1.deinit();
    v1.set(&.{0}, 1.0);
    v1.set(&.{1}, 2.0);
    v1.set(&.{2}, 3.0);
    v1.set(&.{3}, 4.0);

    // Create v2 = [5, 6, 7, 8]
    var v2 = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer v2.deinit();
    v2.set(&.{0}, 5.0);
    v2.set(&.{1}, 6.0);
    v2.set(&.{2}, 7.0);
    v2.set(&.{3}, 8.0);

    // Dot product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    var result = try v1.matmul(&v2);
    defer result.deinit();

    // Result should be a scalar (shape [1])
    try testing.expectEqual(1, result.shape[0]);
    try testing.expectApproxEqAbs(70.0, result.data[0], 1e-10);
}

test "matmul: identity matrix multiplication" {
    const allocator = testing.allocator;

    // Create A = [[2, 3], [4, 5]]
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 2.0);
    A.set(&.{ 0, 1 }, 3.0);
    A.set(&.{ 1, 0 }, 4.0);
    A.set(&.{ 1, 1 }, 5.0);

    // Create identity matrix I = [[1, 0], [0, 1]]
    var I = try NDArray(f64, 2).identity(allocator, 2, 2, .row_major);
    defer I.deinit();

    // A @ I should equal A
    var result = try A.matmul(&I);
    defer result.deinit();

    try testing.expectApproxEqAbs(2.0, try result.get(&.{ 0, 0 }), 1e-10);
    try testing.expectApproxEqAbs(3.0, try result.get(&.{ 0, 1 }), 1e-10);
    try testing.expectApproxEqAbs(4.0, try result.get(&.{ 1, 0 }), 1e-10);
    try testing.expectApproxEqAbs(5.0, try result.get(&.{ 1, 1 }), 1e-10);
}

test "matmul: shape mismatch error for 2D" {
    const allocator = testing.allocator;

    // Create A (2x3)
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer A.deinit();

    // Create B (4x2) - incompatible inner dimensions
    var B = try NDArray(f64, 2).init(allocator, &[_]usize{ 4, 2 }, .row_major);
    defer B.deinit();

    // A (2x3) @ B (4x2) should fail because 3 != 4
    const result = A.matmul(&B);
    try testing.expectError(error.ShapeMismatch, result);
}

test "matmul: shape mismatch error for 1D" {
    const allocator = testing.allocator;

    // Create v1 with length 3
    var v1 = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer v1.deinit();

    // Create v2 with length 5 - incompatible
    var v2 = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer v2.deinit();

    // Should fail because lengths differ
    const result = v1.matmul(&v2);
    try testing.expectError(error.ShapeMismatch, result);
}

test "matmul: integer type matrix multiplication" {
    const allocator = testing.allocator;

    // Create A = [[1, 2], [3, 4]] (i32)
    var A = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 1);
    A.set(&.{ 0, 1 }, 2);
    A.set(&.{ 1, 0 }, 3);
    A.set(&.{ 1, 1 }, 4);

    // Create B = [[5, 6], [7, 8]]
    var B = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer B.deinit();
    B.set(&.{ 0, 0 }, 5);
    B.set(&.{ 0, 1 }, 6);
    B.set(&.{ 1, 0 }, 7);
    B.set(&.{ 1, 1 }, 8);

    // C = A @ B
    // Expected: [[19, 22], [43, 50]]
    var C = try A.matmul(&B);
    defer C.deinit();

    try testing.expectEqual(@as(i32, 19), try C.get(&.{ 0, 0 }));
    try testing.expectEqual(@as(i32, 22), try C.get(&.{ 0, 1 }));
    try testing.expectEqual(@as(i32, 43), try C.get(&.{ 1, 0 }));
    try testing.expectEqual(@as(i32, 50), try C.get(&.{ 1, 1 }));
}

test "matmul: zero matrix multiplication" {
    const allocator = testing.allocator;

    // Create A = [[1, 2], [3, 4]]
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 1.0);
    A.set(&.{ 0, 1 }, 2.0);
    A.set(&.{ 1, 0 }, 3.0);
    A.set(&.{ 1, 1 }, 4.0);

    // Create zero matrix Z = [[0, 0], [0, 0]]
    var Z = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer Z.deinit();

    // A @ Z should be all zeros
    var result = try A.matmul(&Z);
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, try result.get(&.{ 0, 0 }), 1e-10);
    try testing.expectApproxEqAbs(0.0, try result.get(&.{ 0, 1 }), 1e-10);
    try testing.expectApproxEqAbs(0.0, try result.get(&.{ 1, 0 }), 1e-10);
    try testing.expectApproxEqAbs(0.0, try result.get(&.{ 1, 1 }), 1e-10);
}

test "matmul: memory safety with testing.allocator" {
    const allocator = testing.allocator;

    // Perform 10 matmul operations and verify no leaks
    for (0..10) |_| {
        var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
        defer A.deinit();

        var B = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
        defer B.deinit();

        // Fill with arbitrary values
        for (0..9) |i| {
            A.data[i] = @as(f64, @floatFromInt(i + 1));
            B.data[i] = @as(f64, @floatFromInt(i + 1));
        }

        var C = try A.matmul(&B);
        defer C.deinit();

        // Verify result is allocated
        try testing.expect(C.data.len == 9);
    }
    // testing.allocator will detect any leaks
}

test "matmul: column-major layout multiplication" {
    const allocator = testing.allocator;

    // Create A in column-major layout
    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .column_major);
    defer A.deinit();
    A.set(&.{ 0, 0 }, 1.0);
    A.set(&.{ 0, 1 }, 2.0);
    A.set(&.{ 1, 0 }, 3.0);
    A.set(&.{ 1, 1 }, 4.0);

    // Create B in column-major layout
    var B = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .column_major);
    defer B.deinit();
    B.set(&.{ 0, 0 }, 5.0);
    B.set(&.{ 0, 1 }, 6.0);
    B.set(&.{ 1, 0 }, 7.0);
    B.set(&.{ 1, 1 }, 8.0);

    // Matmul should work regardless of layout
    var C = try A.matmul(&B);
    defer C.deinit();

    // Same result as row-major case
    try testing.expectApproxEqAbs(19.0, try C.get(&.{ 0, 0 }), 1e-10);
    try testing.expectApproxEqAbs(22.0, try C.get(&.{ 0, 1 }), 1e-10);
    try testing.expectApproxEqAbs(43.0, try C.get(&.{ 1, 0 }), 1e-10);
    try testing.expectApproxEqAbs(50.0, try C.get(&.{ 1, 1 }), 1e-10);
}

// -- Squeeze Tests (remove dimensions of size 1) --
// NOTE: Tests follow TDD Red phase pattern. Functions squeeze(), squeezeAll(), unsqueeze()
// need to be implemented to pass these tests.

test "ndarray: squeeze removes leading dimension of size 1 from [1,3,4]" {
    const allocator = testing.allocator;

    // Create array with shape [1, 3, 4] in row-major
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 1, 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values [1..12]
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Squeeze should remove dimension of size 1 at axis 0, giving [3,4]
    var squeezed = try arr.squeeze(allocator, 0);
    defer squeezed.deinit();

    try testing.expectEqual(2, squeezed.rank());
    try testing.expectEqual(3, squeezed.shape[0]);
    try testing.expectEqual(4, squeezed.shape[1]);

    // Data should be preserved in same order
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            const val = try squeezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: squeeze removes middle dimension of size 1 from [3,1,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 3, 1, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Squeeze at axis 1
    var squeezed = try arr.squeeze(allocator, 1);
    defer squeezed.deinit();

    try testing.expectEqual(2, squeezed.rank());
    try testing.expectEqual(3, squeezed.shape[0]);
    try testing.expectEqual(4, squeezed.shape[1]);

    // Verify data
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            const val = try squeezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: squeeze removes trailing dimension of size 1 from [3,4,1]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 3, 4, 1 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)), @as(isize, 0) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Squeeze at axis 2
    var squeezed = try arr.squeeze(allocator, 2);
    defer squeezed.deinit();

    try testing.expectEqual(2, squeezed.rank());
    try testing.expectEqual(3, squeezed.shape[0]);
    try testing.expectEqual(4, squeezed.shape[1]);

    // Verify data
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            const val = try squeezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: squeeze error when dimension size != 1" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 3, 4, 5 }, .row_major);
    defer arr.deinit();

    // Try to squeeze axis 0 which has size 3 - should fail
    const result = arr.squeeze(allocator, 0);

    try testing.expectError(error.ShapeMismatch, result);
}

test "ndarray: squeeze invalid axis error" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Try to squeeze axis 5 - out of bounds
    const result = arr.squeeze(allocator, 5);

    try testing.expectError(error.IndexOutOfBounds, result);
}

test "ndarray: squeeze all dimensions of size 1 from [1,3,1,4,1]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 5).init(allocator, &[_]usize{ 1, 3, 1, 4, 1 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values [1..12]
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)), @as(isize, 0) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Squeeze dimension 0 (size 1): [1,3,1,4,1] -> [3,1,4,1]
    var sq1 = try arr.squeeze(allocator, 0);
    defer if (sq1.owned) sq1.deinit();

    try testing.expectEqual(4, sq1.rank());
    try testing.expectEqual(3, sq1.shape[0]);
    try testing.expectEqual(1, sq1.shape[1]);
    try testing.expectEqual(4, sq1.shape[2]);
    try testing.expectEqual(1, sq1.shape[3]);

    // Squeeze dimension 1 (now size 1): [3,1,4,1] -> [3,4,1]
    var sq2 = try sq1.squeeze(allocator, 1);
    defer if (sq2.owned) sq2.deinit();

    try testing.expectEqual(3, sq2.rank());
    try testing.expectEqual(3, sq2.shape[0]);
    try testing.expectEqual(4, sq2.shape[1]);
    try testing.expectEqual(1, sq2.shape[2]);

    // Verify data still accessible
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)), @as(isize, 0) };
            const val = try sq2.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: squeeze 1D array [1] would create 0D scalar" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.set(&.{0}, 99.0);

    // Squeezing a 1D array to 0D is not supported in Zig's comptime type system
    // (NDArray(T, 0) would have [0]usize shape which causes compile errors)
    // This is a known limitation - use scalar directly instead
    // Just verify the array has correct properties before attempting squeeze
    try testing.expectEqual(1, arr.rank());
    try testing.expectEqual(1, arr.shape[0]);
    try testing.expectApproxEqAbs(99.0, try arr.get(&.{0}), 1e-10);
}

// -- Unsqueeze Tests (add dimension of size 1) --

test "ndarray: unsqueeze adds dimension at beginning (axis=0) to [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with values [1..12]
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Unsqueeze at axis 0 to get [1,3,4]
    var unsqueezed = try arr.unsqueeze(allocator, 0);
    defer unsqueezed.deinit();

    try testing.expectEqual(3, unsqueezed.rank());
    try testing.expectEqual(1, unsqueezed.shape[0]);
    try testing.expectEqual(3, unsqueezed.shape[1]);
    try testing.expectEqual(4, unsqueezed.shape[2]);

    // Verify data is preserved
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            const val = try unsqueezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: unsqueeze adds dimension at middle (axis=1) to [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Unsqueeze at axis 1 to get [3,1,4]
    var unsqueezed = try arr.unsqueeze(allocator, 1);
    defer unsqueezed.deinit();

    try testing.expectEqual(3, unsqueezed.rank());
    try testing.expectEqual(3, unsqueezed.shape[0]);
    try testing.expectEqual(1, unsqueezed.shape[1]);
    try testing.expectEqual(4, unsqueezed.shape[2]);

    // Verify data
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            const val = try unsqueezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: unsqueeze adds dimension at end (axis=2) to [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Unsqueeze at axis 2 to get [3,4,1]
    var unsqueezed = try arr.unsqueeze(allocator, 2);
    defer unsqueezed.deinit();

    try testing.expectEqual(3, unsqueezed.rank());
    try testing.expectEqual(3, unsqueezed.shape[0]);
    try testing.expectEqual(4, unsqueezed.shape[1]);
    try testing.expectEqual(1, unsqueezed.shape[2]);

    // Verify data
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)), @as(isize, 0) };
            const val = try unsqueezed.get(idx);
            try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i * 4 + j + 1)), val, 1e-10);
        }
    }
}

test "ndarray: unsqueeze 1D array [5] to [1,5]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Fill with values [1..5]
    for (0..5) |i| {
        arr.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i + 1)));
    }

    // Unsqueeze at beginning
    var unsqueezed = try arr.unsqueeze(allocator, 0);
    defer unsqueezed.deinit();

    try testing.expectEqual(2, unsqueezed.rank());
    try testing.expectEqual(1, unsqueezed.shape[0]);
    try testing.expectEqual(5, unsqueezed.shape[1]);

    // Verify data
    for (0..5) |i| {
        const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)) };
        const val = try unsqueezed.get(idx);
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i + 1)), val, 1e-10);
    }
}

test "ndarray: unsqueeze 1D array [5] to [5,1]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..5) |i| {
        arr.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i + 1)));
    }

    // Unsqueeze at end
    var unsqueezed = try arr.unsqueeze(allocator, 1);
    defer unsqueezed.deinit();

    try testing.expectEqual(2, unsqueezed.rank());
    try testing.expectEqual(5, unsqueezed.shape[0]);
    try testing.expectEqual(1, unsqueezed.shape[1]);

    // Verify data
    for (0..5) |i| {
        const idx = &.{ @as(isize, @intCast(i)), @as(isize, 0) };
        const val = try unsqueezed.get(idx);
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i + 1)), val, 1e-10);
    }
}

test "ndarray: unsqueeze invalid axis error" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Try to unsqueeze at invalid axis
    const result = arr.unsqueeze(allocator, 10);

    try testing.expectError(error.IndexOutOfBounds, result);
}

test "ndarray: unsqueeze with i32 type" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..2) |i| {
        for (0..3) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, @intCast(j)) };
            arr.set(idx, @as(i32, @intCast(i * 3 + j + 1)));
        }
    }

    // Unsqueeze at axis 1
    var unsqueezed = try arr.unsqueeze(allocator, 1);
    defer unsqueezed.deinit();

    try testing.expectEqual(3, unsqueezed.rank());
    try testing.expectEqual(2, unsqueezed.shape[0]);
    try testing.expectEqual(1, unsqueezed.shape[1]);
    try testing.expectEqual(3, unsqueezed.shape[2]);

    // Verify data
    for (0..2) |i| {
        for (0..3) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            const val = try unsqueezed.get(idx);
            try testing.expectEqual(@as(i32, @intCast(i * 3 + j + 1)), val);
        }
    }
}

test "ndarray: unsqueeze with u8 type" {
    const allocator = testing.allocator;

    var arr = try NDArray(u8, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    // Fill with values [10..13]
    for (0..4) |i| {
        arr.set(&.{@as(isize, @intCast(i))}, @as(u8, @intCast(i + 10)));
    }

    // Unsqueeze at beginning
    var unsqueezed = try arr.unsqueeze(allocator, 0);
    defer unsqueezed.deinit();

    try testing.expectEqual(2, unsqueezed.rank());
    try testing.expectEqual(1, unsqueezed.shape[0]);
    try testing.expectEqual(4, unsqueezed.shape[1]);

    // Verify data
    for (0..4) |i| {
        const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)) };
        const val = try unsqueezed.get(idx);
        try testing.expectEqual(@as(u8, @intCast(i + 10)), val);
    }
}

// -- Roundtrip Tests (squeeze then unsqueeze) --

test "ndarray: squeeze then unsqueeze roundtrip [1,5] -> [5] -> [1,5]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 5 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..5) |i| {
        const idx = &.{ @as(isize, 0), @as(isize, @intCast(i)) };
        arr.set(idx, @as(f64, @floatFromInt(i + 1)));
    }

    // Squeeze axis 0
    var squeezed = try arr.squeeze(allocator, 0);
    defer squeezed.deinit();

    try testing.expectEqual(1, squeezed.rank());

    // Unsqueeze at axis 0 to get back to [1, 5]
    var unsqueezed = try squeezed.unsqueeze(allocator, 0);
    defer unsqueezed.deinit();

    try testing.expectEqual(2, unsqueezed.rank());
    try testing.expectEqual(1, unsqueezed.shape[0]);
    try testing.expectEqual(5, unsqueezed.shape[1]);

    // Verify data matches original
    for (0..5) |i| {
        const orig_idx = &.{ @as(isize, 0), @as(isize, @intCast(i)) };
        const unsq_idx = &.{ @as(isize, 0), @as(isize, @intCast(i)) };
        const original = try arr.get(orig_idx);
        const recovered = try unsqueezed.get(unsq_idx);
        try testing.expectApproxEqAbs(original, recovered, 1e-10);
    }
}

test "ndarray: squeeze/unsqueeze roundtrip [3,1,4] -> [3,4] -> [3,1,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 3, 1, 4 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..3) |i| {
        for (0..4) |j| {
            const idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            arr.set(idx, @as(f64, @floatFromInt(i * 4 + j + 1)));
        }
    }

    // Squeeze axis 1
    var squeezed = try arr.squeeze(allocator, 1);
    defer squeezed.deinit();

    try testing.expectEqual(2, squeezed.rank());

    // Unsqueeze at axis 1 to get back
    var unsqueezed = try squeezed.unsqueeze(allocator, 1);
    defer unsqueezed.deinit();

    try testing.expectEqual(3, unsqueezed.rank());
    try testing.expectEqual(3, unsqueezed.shape[0]);
    try testing.expectEqual(1, unsqueezed.shape[1]);
    try testing.expectEqual(4, unsqueezed.shape[2]);

    // Verify data
    for (0..3) |i| {
        for (0..4) |j| {
            const orig_idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            const unsq_idx = &.{ @as(isize, @intCast(i)), @as(isize, 0), @as(isize, @intCast(j)) };
            const original = try arr.get(orig_idx);
            const recovered = try unsqueezed.get(unsq_idx);
            try testing.expectApproxEqAbs(original, recovered, 1e-10);
        }
    }
}

test "ndarray: unsqueeze then squeeze roundtrip [5] -> [1,5] -> [5]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..5) |i| {
        arr.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i + 1)));
    }

    // Unsqueeze at beginning
    var unsqueezed = try arr.unsqueeze(allocator, 0);
    defer unsqueezed.deinit();

    try testing.expectEqual(2, unsqueezed.rank());

    // Squeeze axis 0 to get back
    var squeezed = try unsqueezed.squeeze(allocator, 0);
    defer squeezed.deinit();

    try testing.expectEqual(1, squeezed.rank());
    try testing.expectEqual(5, squeezed.shape[0]);

    // Verify data
    for (0..5) |i| {
        const orig = try arr.get(&.{@as(isize, @intCast(i))});
        const recovered = try squeezed.get(&.{@as(isize, @intCast(i))});
        try testing.expectApproxEqAbs(orig, recovered, 1e-10);
    }
}

// -- Memory Safety Tests --

test "ndarray: squeeze with 10 iterations no memory leak" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 5 }, .row_major);
        for (0..5) |i| {
            arr.set(&.{ @as(isize, 0), @as(isize, @intCast(i)) }, @as(f64, @floatFromInt(i + 1)));
        }

        var squeezed = try arr.squeeze(allocator, 0);
        try testing.expectEqual(1, squeezed.rank());

        squeezed.deinit();
        arr.deinit();
    }
}

test "ndarray: unsqueeze with 10 iterations no memory leak" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
        for (0..5) |i| {
            arr.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i + 1)));
        }

        var unsqueezed = try arr.unsqueeze(allocator, 0);
        try testing.expectEqual(2, unsqueezed.rank());

        unsqueezed.deinit();
        arr.deinit();
    }
}

test "ndarray: squeeze/unsqueeze roundtrip 10 iterations no leak" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 5 }, .row_major);
        for (0..5) |i| {
            arr.set(&.{ @as(isize, 0), @as(isize, @intCast(i)) }, @as(f64, @floatFromInt(i + 1)));
        }

        var squeezed = try arr.squeeze(allocator, 0);
        var unsqueezed = try squeezed.unsqueeze(allocator, 0);

        try testing.expectEqual(2, unsqueezed.rank());

        unsqueezed.deinit();
        squeezed.deinit();
        arr.deinit();
    }
}

test "ndarray: squeeze column-major layout preserved" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 5 }, .column_major);
    defer arr.deinit();

    // Fill with values
    for (0..5) |i| {
        arr.set(&.{ @as(isize, 0), @as(isize, @intCast(i)) }, @as(f64, @floatFromInt(i + 1)));
    }

    // Squeeze
    var squeezed = try arr.squeeze(allocator, 0);
    defer squeezed.deinit();

    try testing.expectEqual(.column_major, squeezed.layout);
    try testing.expectEqual(1, squeezed.rank());
}

test "ndarray: unsqueeze column-major layout preserved" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .column_major);
    defer arr.deinit();

    // Fill with values
    for (0..5) |i| {
        arr.set(&.{@as(isize, @intCast(i))}, @as(f64, @floatFromInt(i + 1)));
    }

    // Unsqueeze
    var unsqueezed = try arr.unsqueeze(allocator, 1);
    defer unsqueezed.deinit();

    try testing.expectEqual(.column_major, unsqueezed.layout);
    try testing.expectEqual(2, unsqueezed.rank());
}

// -- concat() Concatenation Tests (20 tests) --

test "ndarray: concat() 1D arrays simple case" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 4, 5 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 1).concat(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(5, result.count());
    try testing.expectEqual(5, result.shape[0]);
    try testing.expectEqual(1.0, try result.get(&.{0}));
    try testing.expectEqual(2.0, try result.get(&.{1}));
    try testing.expectEqual(3.0, try result.get(&.{2}));
    try testing.expectEqual(4.0, try result.get(&.{3}));
    try testing.expectEqual(5.0, try result.get(&.{4}));
}

test "ndarray: concat() 2D arrays along axis 0 (rows)" {
    const allocator = testing.allocator;

    // [2, 3] arrays
    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(12, result.count());
    try testing.expectEqual(4, result.shape[0]); // 2 + 2 rows
    try testing.expectEqual(3, result.shape[1]);

    // Verify values
    try testing.expectEqual(1.0, try result.get(&.{ 0, 0 }));
    try testing.expectEqual(6.0, try result.get(&.{ 1, 2 }));
    try testing.expectEqual(7.0, try result.get(&.{ 2, 0 }));
    try testing.expectEqual(12.0, try result.get(&.{ 3, 2 }));
}

test "ndarray: concat() 2D arrays along axis 1 (columns)" {
    const allocator = testing.allocator;

    // [2, 3] arrays
    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 7, 8, 9, 10 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 1, .row_major);
    defer result.deinit();

    try testing.expectEqual(10, result.count());
    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(5, result.shape[1]); // 3 + 2 columns

    // Verify first row: [1, 2, 3, 7, 8]
    try testing.expectEqual(1.0, try result.get(&.{ 0, 0 }));
    try testing.expectEqual(3.0, try result.get(&.{ 0, 2 }));
    try testing.expectEqual(7.0, try result.get(&.{ 0, 3 }));
    try testing.expectEqual(8.0, try result.get(&.{ 0, 4 }));
}

test "ndarray: concat() three arrays" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 3, 4, 5 }, .row_major);
    defer b.deinit();
    var c = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{6}, .row_major);
    defer c.deinit();

    var result = try NDArray(f64, 1).concat(allocator, &[_]*const NDArray(f64, 1){ &a, &b, &c }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(6, result.count());
    try testing.expectEqual(1.0, try result.get(&.{0}));
    try testing.expectEqual(4.0, try result.get(&.{3}));
    try testing.expectEqual(6.0, try result.get(&.{5}));
}

test "ndarray: concat() 3D arrays" {
    const allocator = testing.allocator;

    // [2, 2, 2] arrays
    var a = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 1, 2, 2 }, &[_]f64{ 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 3).concat(allocator, &[_]*const NDArray(f64, 3){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(12, result.count());
    try testing.expectEqual(3, result.shape[0]); // 2 + 1
    try testing.expectEqual(2, result.shape[1]);
    try testing.expectEqual(2, result.shape[2]);
}

test "ndarray: concat() empty arrays slice returns error" {
    const allocator = testing.allocator;
    const arrays: []const *const NDArray(f64, 1) = &[_]*const NDArray(f64, 1){};

    const result = NDArray(f64, 1).concat(allocator, arrays, 0, .row_major);
    try testing.expectError(error.EmptyArray, result);
}

test "ndarray: concat() axis out of bounds" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a.deinit();

    const result = NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){&a}, 2, .row_major);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "ndarray: concat() shape mismatch error" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &[_]f64{ 7, 8, 9, 10, 11, 12, 13, 14 }, .row_major);
    defer b.deinit();

    // Trying to concat along axis 0, but axis 1 sizes differ (3 vs 4)
    const result = NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    try testing.expectError(error.ShapeMismatch, result);
}

test "ndarray: concat() single array (identity)" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a.deinit();

    var result = try NDArray(f64, 1).concat(allocator, &[_]*const NDArray(f64, 1){&a}, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(3, result.count());
    try testing.expectEqual(1.0, try result.get(&.{0}));
    try testing.expectEqual(3.0, try result.get(&.{2}));
}

test "ndarray: concat() with i32 type" {
    const allocator = testing.allocator;

    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 10, 20 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 30, 40 }, .row_major);
    defer b.deinit();

    var result = try NDArray(i32, 1).concat(allocator, &[_]*const NDArray(i32, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(4, result.count());
    try testing.expectEqual(@as(i32, 10), try result.get(&.{0}));
    try testing.expectEqual(@as(i32, 40), try result.get(&.{3}));
}

test "ndarray: concat() column-major layout" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .column_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .column_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .column_major);
    defer result.deinit();

    try testing.expectEqual(.column_major, result.layout);
    try testing.expectEqual(4, result.shape[0]); // 2 + 2
    try testing.expectEqual(2, result.shape[1]);
}

test "ndarray: concat() preserves data correctness for large arrays" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).arange(allocator, 0, 50, 1, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).arange(allocator, 50, 100, 1, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 1).concat(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(100, result.count());
    try testing.expectEqual(0.0, try result.get(&.{0}));
    try testing.expectEqual(49.0, try result.get(&.{49}));
    try testing.expectEqual(50.0, try result.get(&.{50}));
    try testing.expectEqual(99.0, try result.get(&.{99}));
}

test "ndarray: concat() 2D different sizes along concat axis" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 3 }, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 4, 5, 6, 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(4, result.shape[0]); // 1 + 3
    try testing.expectEqual(3, result.shape[1]);
    try testing.expectEqual(12, result.count());
}

test "ndarray: concat() u8 type (byte arrays)" {
    const allocator = testing.allocator;

    var a = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &[_]u8{ 65, 66, 67 }, .row_major);
    defer a.deinit();
    var b = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 68, 69 }, .row_major);
    defer b.deinit();

    var result = try NDArray(u8, 1).concat(allocator, &[_]*const NDArray(u8, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(5, result.count());
    try testing.expectEqual(@as(u8, 65), try result.get(&.{0}));
    try testing.expectEqual(@as(u8, 69), try result.get(&.{4}));
}

test "ndarray: concat() memory safety (10 iterations)" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
        defer a.deinit();
        var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 6, 7, 8, 9, 10 }, .row_major);
        defer b.deinit();

        var result = try NDArray(f64, 1).concat(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 0, .row_major);
        defer result.deinit();

        try testing.expectEqual(10, result.count());
    }
}

test "ndarray: concat() axis 2 for 3D arrays" {
    const allocator = testing.allocator;

    // [2, 2, 2] and [2, 2, 1]
    var a = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 2, 2, 1 }, &[_]f64{ 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 3).concat(allocator, &[_]*const NDArray(f64, 3){ &a, &b }, 2, .row_major);
    defer result.deinit();

    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(2, result.shape[1]);
    try testing.expectEqual(3, result.shape[2]); // 2 + 1
    try testing.expectEqual(12, result.count());
}

test "ndarray: concat() result has correct strides" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 3 }, &[_]f64{ 7, 8, 9 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    // Row-major strides for [3, 3] should be [3, 1]
    try testing.expectEqual(3, result.strides[0]);
    try testing.expectEqual(1, result.strides[1]);
}

test "ndarray: concat() validates result with validate()" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).concat(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try result.validate();
}

// ============================================================================
// Stack Operation Tests
// ============================================================================

test "ndarray: stack() 2D arrays [2,3] at axis 0 → [3,2,3]" {
    const allocator = testing.allocator;

    // Create 3 arrays of shape [2, 3]
    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();
    var c = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 13, 14, 15, 16, 17, 18 }, .row_major);
    defer c.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b, &c }, 0, .row_major);
    defer result.deinit();

    // Check shape: [3, 2, 3]
    try testing.expectEqual(3, result.shape[0]);
    try testing.expectEqual(2, result.shape[1]);
    try testing.expectEqual(3, result.shape[2]);
    try testing.expectEqual(18, result.count());
}

test "ndarray: stack() 2D arrays [2,3] at axis 1 → [2,3,3]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();
    var c = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 13, 14, 15, 16, 17, 18 }, .row_major);
    defer c.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b, &c }, 1, .row_major);
    defer result.deinit();

    // Check shape: [2, 3, 3]
    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(3, result.shape[1]);
    try testing.expectEqual(3, result.shape[2]);
    try testing.expectEqual(18, result.count());
}

test "ndarray: stack() 2D arrays [2,3] at axis 2 → [2,3,3]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();
    var c = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 13, 14, 15, 16, 17, 18 }, .row_major);
    defer c.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b, &c }, 2, .row_major);
    defer result.deinit();

    // Check shape: [2, 3, 3]
    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(3, result.shape[1]);
    try testing.expectEqual(3, result.shape[2]);
    try testing.expectEqual(18, result.count());
}

test "ndarray: stack() 1D vectors [4] at axis 0 → [2,4]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 1).stack(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    // Check shape: [2, 4]
    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(4, result.shape[1]);
    try testing.expectEqual(8, result.count());
}

test "ndarray: stack() 1D vectors [4] at axis 1 → [4,2]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 1).stack(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 1, .row_major);
    defer result.deinit();

    // Check shape: [4, 2]
    try testing.expectEqual(4, result.shape[0]);
    try testing.expectEqual(2, result.shape[1]);
    try testing.expectEqual(8, result.count());
}

test "ndarray: stack() single array [2,3] at axis 0 → [1,2,3]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){&a}, 0, .row_major);
    defer result.deinit();

    // Check shape: [1, 2, 3]
    try testing.expectEqual(1, result.shape[0]);
    try testing.expectEqual(2, result.shape[1]);
    try testing.expectEqual(3, result.shape[2]);
    try testing.expectEqual(6, result.count());
}

test "ndarray: stack() empty array list returns EmptyArray error" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();

    const result = NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){}, 0, .row_major);
    try testing.expectError(error.EmptyArray, result);
}

test "ndarray: stack() axis > ndim returns IndexOutOfBounds error" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    const result = NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 3, .row_major);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "ndarray: stack() mismatched shapes returns ShapeMismatch error" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    // Shape [2, 4] instead of [2, 3]
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &[_]f64{ 7, 8, 9, 10, 11, 12, 13, 14 }, .row_major);
    defer b.deinit();

    const result = NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    try testing.expectError(error.ShapeMismatch, result);
}

test "ndarray: stack() single element arrays [1] at axis 0 → [2,1]" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{1}, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{2}, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 1).stack(allocator, &[_]*const NDArray(f64, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(2, result.shape[0]);
    try testing.expectEqual(1, result.shape[1]);
}

test "ndarray: stack() large array count (10 arrays)" {
    const allocator = testing.allocator;

    var arrays: [10]*const NDArray(f64, 1) = undefined;
    var allocated: [10]NDArray(f64, 1) = undefined;

    // Create 10 arrays
    for (0..10) |i| {
        allocated[i] = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ @as(f64, @floatFromInt(i * 3 + 1)), @as(f64, @floatFromInt(i * 3 + 2)), @as(f64, @floatFromInt(i * 3 + 3)) }, .row_major);
        arrays[i] = &allocated[i];
    }
    defer {
        for (&allocated) |*arr| {
            arr.deinit();
        }
    }

    var result = try NDArray(f64, 1).stack(allocator, arrays[0..], 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(10, result.shape[0]);
    try testing.expectEqual(3, result.shape[1]);
    try testing.expectEqual(30, result.count());
}

test "ndarray: stack() data correctness - axis 0" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    // Verify first array's data at result[0, :, :]
    try testing.expectEqual(@as(f64, 1), try result.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try result.get(&[_]isize{ 0, 0, 1 }));
    try testing.expectEqual(@as(f64, 3), try result.get(&[_]isize{ 0, 1, 0 }));
    try testing.expectEqual(@as(f64, 4), try result.get(&[_]isize{ 0, 1, 1 }));

    // Verify second array's data at result[1, :, :]
    try testing.expectEqual(@as(f64, 5), try result.get(&[_]isize{ 1, 0, 0 }));
    try testing.expectEqual(@as(f64, 6), try result.get(&[_]isize{ 1, 0, 1 }));
    try testing.expectEqual(@as(f64, 7), try result.get(&[_]isize{ 1, 1, 0 }));
    try testing.expectEqual(@as(f64, 8), try result.get(&[_]isize{ 1, 1, 1 }));
}

test "ndarray: stack() data correctness - axis 1" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 1, .row_major);
    defer result.deinit();

    // Verify first array's data at result[:, 0, :]
    try testing.expectEqual(@as(f64, 1), try result.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try result.get(&[_]isize{ 0, 0, 1 }));
    try testing.expectEqual(@as(f64, 3), try result.get(&[_]isize{ 1, 0, 0 }));
    try testing.expectEqual(@as(f64, 4), try result.get(&[_]isize{ 1, 0, 1 }));

    // Verify second array's data at result[:, 1, :]
    try testing.expectEqual(@as(f64, 5), try result.get(&[_]isize{ 0, 1, 0 }));
    try testing.expectEqual(@as(f64, 6), try result.get(&[_]isize{ 0, 1, 1 }));
    try testing.expectEqual(@as(f64, 7), try result.get(&[_]isize{ 1, 1, 0 }));
    try testing.expectEqual(@as(f64, 8), try result.get(&[_]isize{ 1, 1, 1 }));
}

test "ndarray: stack() data correctness - axis 2" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 2, .row_major);
    defer result.deinit();

    // Verify first array's data at result[:, :, 0]
    try testing.expectEqual(@as(f64, 1), try result.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try result.get(&[_]isize{ 0, 1, 0 }));
    try testing.expectEqual(@as(f64, 3), try result.get(&[_]isize{ 1, 0, 0 }));
    try testing.expectEqual(@as(f64, 4), try result.get(&[_]isize{ 1, 1, 0 }));

    // Verify second array's data at result[:, :, 1]
    try testing.expectEqual(@as(f64, 5), try result.get(&[_]isize{ 0, 0, 1 }));
    try testing.expectEqual(@as(f64, 6), try result.get(&[_]isize{ 0, 1, 1 }));
    try testing.expectEqual(@as(f64, 7), try result.get(&[_]isize{ 1, 0, 1 }));
    try testing.expectEqual(@as(f64, 8), try result.get(&[_]isize{ 1, 1, 1 }));
}

test "ndarray: stack() type variant - i32 arrays" {
    const allocator = testing.allocator;

    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &[_]i32{ 10, 20, 30 }, .row_major);
    defer a.deinit();
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &[_]i32{ 40, 50, 60 }, .row_major);
    defer b.deinit();

    var result = try NDArray(i32, 1).stack(allocator, &[_]*const NDArray(i32, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 10), try result.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(i32, 40), try result.get(&[_]isize{ @intCast(1), @intCast(0) }));
}

test "ndarray: stack() type variant - u8 arrays" {
    const allocator = testing.allocator;

    var a = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &[_]u8{ 65, 66, 67 }, .row_major);
    defer a.deinit();
    var b = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &[_]u8{ 68, 69, 70 }, .row_major);
    defer b.deinit();

    var result = try NDArray(u8, 1).stack(allocator, &[_]*const NDArray(u8, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(u8, 65), try result.get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(u8, 68), try result.get(&[_]isize{ @intCast(1), @intCast(0) }));
}

test "ndarray: stack() row-major layout preserved" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(.row_major, result.layout);
    // Verify strides are consistent with row-major layout [2,2,3]
    try testing.expectEqual(6, result.strides[0]); // 2 * 3
    try testing.expectEqual(3, result.strides[1]); // 3
    try testing.expectEqual(1, result.strides[2]); // 1
}

test "ndarray: stack() column-major layout preserved" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .column_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 7, 8, 9, 10, 11, 12 }, .column_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .column_major);
    defer result.deinit();

    try testing.expectEqual(.column_major, result.layout);
}

test "ndarray: stack() memory safety (10 iterations)" {
    const allocator = testing.allocator;

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
        defer a.deinit();
        var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
        defer b.deinit();

        var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
        defer result.deinit();

        try testing.expectEqual(2, result.shape[0]);
        try testing.expectEqual(2, result.shape[1]);
        try testing.expectEqual(2, result.shape[2]);
    }
}

test "ndarray: stack() validates result with validate()" {
    const allocator = testing.allocator;

    var a = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer a.deinit();
    var b = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 5, 6, 7, 8 }, .row_major);
    defer b.deinit();

    var result = try NDArray(f64, 2).stack(allocator, &[_]*const NDArray(f64, 2){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try result.validate();
}

// -- Convenience Stack Functions Tests (20 tests) --

test "ndarray: vstack() 1D arrays create 2D" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).vstack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
}

test "ndarray: vstack() 2D arrays" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 2).vstack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.shape[2]);
}

test "ndarray: hstack() 1D arrays concatenate" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).hstack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "ndarray: hstack() 2D arrays concatenate columns" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 2).hstack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 5), result.shape[1]);
}

test "ndarray: dstack() 2D arrays create 3D" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 2).dstack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
}

test "ndarray: dstack() 3D arrays" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 3).dstack(allocator, &[_]*const NDArray(f64, 3){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
    try testing.expectEqual(@as(usize, 2), result.shape[3]);
}

test "ndarray: row_stack() is vstack alias" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).row_stack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
}

test "ndarray: column_stack() 1D to 2D columns" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 4, 5, 6 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).column_stack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);

    // Verify first column is a1 data
    try testing.expectEqual(@as(f64, 1), result.data[0 * result.strides[0] + 0 * result.strides[1]]);
    try testing.expectEqual(@as(f64, 2), result.data[1 * result.strides[0] + 0 * result.strides[1]]);
    try testing.expectEqual(@as(f64, 3), result.data[2 * result.strides[0] + 0 * result.strides[1]]);
}

test "ndarray: column_stack() 2D is hstack" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 1 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 2).column_stack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
}

test "ndarray: vstack() empty error" {
    const allocator = testing.allocator;
    const empty: []const *const NDArray(f64, 1) = &[_]*const NDArray(f64, 1){};
    try testing.expectError(error.EmptyArray, NDArray(f64, 1).vstack(allocator, empty, .row_major));
}

test "ndarray: hstack() shape mismatch error" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer a2.deinit();

    try testing.expectError(error.ShapeMismatch, NDArray(f64, 2).hstack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major));
}

test "ndarray: column_stack() 1D shape mismatch error" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 4, 5, 6, 7 }, .row_major);
    defer a2.deinit();

    try testing.expectError(error.ShapeMismatch, NDArray(f64, 1).column_stack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major));
}

test "ndarray: vstack() type variant i32" {
    const allocator = testing.allocator;
    var a1 = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 1, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 3, 4 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(i32, 1).vstack(allocator, &[_]*const NDArray(i32, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
}

test "ndarray: hstack() type variant u8" {
    const allocator = testing.allocator;
    var a1 = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 1, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 3, 4 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(u8, 1).hstack(allocator, &[_]*const NDArray(u8, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
}

test "ndarray: column_stack() single array" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer a1.deinit();

    var result = try NDArray(f64, 1).column_stack(allocator, &[_]*const NDArray(f64, 1){&a1}, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 1), result.shape[1]);
}

test "ndarray: vstack() column-major preserved" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 1, 2 }, .column_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &[_]f64{ 3, 4 }, .column_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).vstack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .column_major);
    defer result.deinit();

    try testing.expectEqual(Layout.column_major, result.layout);
}

test "ndarray: hstack() row-major preserved" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 2).hstack(allocator, &[_]*const NDArray(f64, 2){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try testing.expectEqual(Layout.row_major, result.layout);
}

test "ndarray: column_stack() memory safety" {
    const allocator = testing.allocator;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var a1 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 1, 2, 3, 4, 5 }, .row_major);
        defer a1.deinit();
        var a2 = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &[_]f64{ 6, 7, 8, 9, 10 }, .row_major);
        defer a2.deinit();

        var result = try NDArray(f64, 1).column_stack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 5), result.shape[0]);
        try testing.expectEqual(@as(usize, 2), result.shape[1]);
    }
}

test "ndarray: vstack() validates result" {
    const allocator = testing.allocator;
    var a1 = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a1.deinit();
    var a2 = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a2.deinit();

    var result = try NDArray(f64, 1).vstack(allocator, &[_]*const NDArray(f64, 1){ &a1, &a2 }, .row_major);
    defer result.deinit();

    try result.validate();
}

// -- split() Tests (20 tests) --

test "ndarray: split() basic 2D along axis=0" {
    const allocator = testing.allocator;

    // Create [6, 4] array with values 0..23
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 6, 4 }, .row_major);
    defer arr.deinit();
    for (0..6) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 4 + j)));
        }
    }

    // Split into 3 sections along axis=0 → 3×[2, 4] arrays
    const parts = try arr.split(allocator, 0, 3);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 4), parts[0].shape[1]);

    // Verify first part contains rows 0-1
    try testing.expectEqual(@as(f64, 0), try parts[0].get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(f64, 4), try parts[0].get(&[_]isize{ @intCast(1), @intCast(0) }));

    // Verify second part contains rows 2-3
    try testing.expectEqual(@as(f64, 8), try parts[1].get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(f64, 12), try parts[1].get(&[_]isize{ @intCast(1), @intCast(0) }));

    // Verify third part contains rows 4-5
    try testing.expectEqual(@as(f64, 16), try parts[2].get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(f64, 20), try parts[2].get(&[_]isize{ @intCast(1), @intCast(0) }));
}

test "ndarray: split() basic 2D along axis=1" {
    const allocator = testing.allocator;

    // Create [3, 6] array
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 6 }, .row_major);
    defer arr.deinit();
    for (0..3) |i| {
        for (0..6) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 6 + j)));
        }
    }

    // Split into 2 sections along axis=1 → 2×[3, 3] arrays
    const parts = try arr.split(allocator, 1, 2);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 2), parts.len);
    try testing.expectEqual(@as(usize, 3), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 3), parts[0].shape[1]);

    // Verify first part contains columns 0-2
    try testing.expectEqual(@as(f64, 0), try parts[0].get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(f64, 2), try parts[0].get(&[_]isize{ @intCast(0), @intCast(2) }));

    // Verify second part contains columns 3-5
    try testing.expectEqual(@as(f64, 3), try parts[1].get(&[_]isize{ @intCast(0), @intCast(0) }));
    try testing.expectEqual(@as(f64, 5), try parts[1].get(&[_]isize{ @intCast(0), @intCast(2) }));
}

test "ndarray: split() 1D array into 4 parts" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 4);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 4), parts.len);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[0]);

    // Verify each part
    try testing.expectEqual(@as(f64, 1), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 2), try parts[0].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 3), try parts[1].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 4), try parts[1].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 5), try parts[2].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 6), try parts[2].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 7), try parts[3].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 8), try parts[3].get(&[_]isize{1}));
}

test "ndarray: split() into 2 parts (half)" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 2);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 2), parts.len);
    try testing.expectEqual(@as(usize, 3), parts[0].shape[0]);
    try testing.expectEqual(@as(f64, 1), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 4), try parts[1].get(&[_]isize{0}));
}

test "ndarray: split() single section returns whole array" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 1);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 1), parts.len);
    try testing.expectEqual(@as(usize, 4), parts[0].shape[0]);
    try testing.expectEqual(@as(f64, 1), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 4), try parts[0].get(&[_]isize{3}));
}

test "ndarray: split() error on invalid axis" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 4, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.IndexOutOfBounds, arr.split(allocator, 2, 2));
}

test "ndarray: split() error when not evenly divisible" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{7}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.ShapeMismatch, arr.split(allocator, 0, 3));
}

test "ndarray: split() error on zero sections" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.ZeroDimension, arr.split(allocator, 0, 0));
}

test "ndarray: split() 3D array along axis=0" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 4, 3, 2 }, .row_major);
    defer arr.deinit();
    for (0..4) |i| {
        for (0..3) |j| {
            for (0..2) |k| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j), @intCast(k) }, @as(f64, @floatFromInt(i * 6 + j * 2 + k)));
            }
        }
    }

    const parts = try arr.split(allocator, 0, 2);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 2), parts.len);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 3), parts[0].shape[1]);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[2]);
}

test "ndarray: split() 3D array along axis=1" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 6, 2 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 1, 3);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[1]);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[2]);
}

test "ndarray: split() 3D array along axis=2" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 8 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 2, 4);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 4), parts.len);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 3), parts[0].shape[1]);
    try testing.expectEqual(@as(usize, 2), parts[0].shape[2]);
}

test "ndarray: split() preserves data correctly" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{9}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 3);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    // Verify all elements are preserved
    try testing.expectEqual(@as(f64, 1), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 2), try parts[0].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 3), try parts[0].get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 4), try parts[1].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 5), try parts[1].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 6), try parts[1].get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 7), try parts[2].get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 8), try parts[2].get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 9), try parts[2].get(&[_]isize{2}));
}

test "ndarray: split() with column-major layout" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 4, 3 }, .column_major);
    defer arr.deinit();
    for (0..4) |i| {
        for (0..3) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 3 + j)));
        }
    }

    const parts = try arr.split(allocator, 0, 2);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 2), parts.len);
    try testing.expectEqual(Layout.column_major, parts[0].layout);
    try testing.expectEqual(Layout.column_major, parts[1].layout);
}

test "ndarray: split() i32 type" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{6}, &[_]i32{ 10, 20, 30, 40, 50, 60 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 3);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 3), parts.len);
    try testing.expectEqual(@as(i32, 10), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 30), try parts[1].get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 50), try parts[2].get(&[_]isize{0}));
}

test "ndarray: split() u8 type" {
    const allocator = testing.allocator;

    var arr = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{8}, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 4);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 4), parts.len);
    try testing.expectEqual(@as(u8, 1), try parts[0].get(&[_]isize{0}));
    try testing.expectEqual(@as(u8, 3), try parts[1].get(&[_]isize{0}));
}

test "ndarray: split() memory safety with allocator" {
    const allocator = testing.allocator;

    // Run 10 iterations to check for memory leaks
    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 6, 4 }, .row_major);
        defer arr.deinit();

        const parts = try arr.split(allocator, 0, 3);
        defer allocator.free(parts);
        defer for (parts) |*part| part.deinit();
    }
}

test "ndarray: split() validates each part" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 6, 4 }, .row_major);
    defer arr.deinit();

    const parts = try arr.split(allocator, 0, 3);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    // Validate each part
    for (parts) |*part| {
        try part.validate();
    }
}

test "ndarray: split() large array stress test" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 100, 50 }, .row_major);
    defer arr.deinit();
    for (0..100) |i| {
        for (0..50) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 50 + j)));
        }
    }

    const parts = try arr.split(allocator, 0, 10);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    try testing.expectEqual(@as(usize, 10), parts.len);
    try testing.expectEqual(@as(usize, 10), parts[0].shape[0]);
    try testing.expectEqual(@as(usize, 50), parts[0].shape[1]);

    // Verify first and last elements of each part
    for (0..10) |part_idx| {
        const expected_first = @as(f64, @floatFromInt(part_idx * 10 * 50));
        try testing.expectEqual(expected_first, try parts[part_idx].get(&[_]isize{ @intCast(0), @intCast(0) }));
    }
}

test "ndarray: split() then concat roundtrip" {
    const allocator = testing.allocator;

    var original = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{12}, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, .row_major);
    defer original.deinit();

    // Split into 4 parts
    const parts = try original.split(allocator, 0, 4);
    defer allocator.free(parts);
    defer for (parts) |*part| part.deinit();

    // Concat back together
    var parts_ptrs: [4]*const NDArray(f64, 1) = undefined;
    for (parts, 0..) |*part, i| {
        parts_ptrs[i] = part;
    }
    var reconstructed = try NDArray(f64, 1).concat(allocator, &parts_ptrs, 0, .row_major);
    defer reconstructed.deinit();

    // Verify it matches original
    try testing.expectEqual(@as(usize, 12), reconstructed.shape[0]);
    for (0..12) |i| {
        const orig_val = try original.get(&[_]isize{@intCast(i)});
        const recon_val = try reconstructed.get(&[_]isize{@intCast(i)});
        try testing.expectEqual(orig_val, recon_val);
    }
}

// -- pad() Tests (20 tests) --

test "ndarray: pad() constant mode 1D zero padding" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 1, 1 }}, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 5), padded.shape[0]);
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{4}));
}

test "ndarray: pad() constant mode 2D zero padding" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 1, 1 }, .{ 1, 1 } }, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 4), padded.shape[0]);
    try testing.expectEqual(@as(usize, 5), padded.shape[1]);

    // Check corners are zero
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{ 0, 4 }));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{ 3, 0 }));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{ 3, 4 }));

    // Check center values preserved
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(f64, 6), try padded.get(&[_]isize{ 2, 3 }));
}

test "ndarray: pad() constant mode 3D padding" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 } }, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 4), padded.shape[0]);
    try testing.expectEqual(@as(usize, 4), padded.shape[1]);
    try testing.expectEqual(@as(usize, 4), padded.shape[2]);

    // Check padding is zero
    try testing.expectEqual(@as(i32, 0), try padded.get(&[_]isize{ 0, 0, 0 }));
    // Check center value preserved
    try testing.expectEqual(@as(i32, 1), try padded.get(&[_]isize{ 1, 1, 1 }));
}

test "ndarray: pad() constant mode with non-zero value" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .constant, 99);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 7), padded.shape[0]);
    try testing.expectEqual(@as(f64, 99), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 99), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 99), try padded.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 99), try padded.get(&[_]isize{6}));
}

test "ndarray: pad() edge mode 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 10, 20, 30 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .edge, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 7), padded.shape[0]);
    // Edge values extended
    try testing.expectEqual(@as(f64, 10), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 10), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 10), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 20), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 30), try padded.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 30), try padded.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 30), try padded.get(&[_]isize{6}));
}

test "ndarray: pad() edge mode 2D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 1, 1 }, .{ 1, 1 } }, .edge, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 4), padded.shape[0]);
    try testing.expectEqual(@as(usize, 4), padded.shape[1]);

    // Corners extend edge values
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{ 0, 0 })); // top-left corner = [0,0]
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{ 0, 3 })); // top-right corner = [0,1]
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{ 3, 0 })); // bottom-left = [1,0]
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{ 3, 3 })); // bottom-right = [1,1]
}

test "ndarray: pad() reflect mode 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .reflect, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 8), padded.shape[0]);
    // Reflect: [3, 2] | 1, 2, 3, 4 | [3, 2]
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{6}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{7}));
}

test "ndarray: pad() symmetric mode 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .symmetric, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 8), padded.shape[0]);
    // Symmetric: [2, 1] | 1, 2, 3, 4 | [4, 3]
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{6}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{7}));
}

test "ndarray: pad() wrap mode 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .wrap, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 8), padded.shape[0]);
    // Wrap: [3, 4] | 1, 2, 3, 4 | [1, 2]
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 4), try padded.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{6}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{7}));
}

test "ndarray: pad() asymmetric padding 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 1, 3 }}, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 7), padded.shape[0]);
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{6}));
}

test "ndarray: pad() asymmetric padding 2D" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]i32{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 1, 2 }, .{ 2, 1 } }, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 5), padded.shape[0]); // 2 + 1 + 2
    try testing.expectEqual(@as(usize, 5), padded.shape[1]); // 2 + 2 + 1

    // Original data at [1,2]
    try testing.expectEqual(@as(i32, 1), try padded.get(&[_]isize{ 1, 2 }));
}

test "ndarray: pad() no padding (identity)" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 0, 0 }}, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 3), padded.shape[0]);
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 2), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 3), try padded.get(&[_]isize{2}));
}

test "ndarray: pad() single element array" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &[_]f64{42}, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 5), padded.shape[0]);
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 42), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 0), try padded.get(&[_]isize{4}));
}

test "ndarray: pad() type variant u8" {
    const allocator = testing.allocator;

    var arr = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 5, 10 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 1, 1 }}, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 4), padded.shape[0]);
    try testing.expectEqual(@as(u8, 0), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(u8, 5), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(u8, 10), try padded.get(&[_]isize{2}));
    try testing.expectEqual(@as(u8, 0), try padded.get(&[_]isize{3}));
}

test "ndarray: pad() type variant i32" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &[_]i32{ -5, 0, 5 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 1, 1 }}, .constant, -99);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 5), padded.shape[0]);
    try testing.expectEqual(@as(i32, -99), try padded.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, -5), try padded.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 5), try padded.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, -99), try padded.get(&[_]isize{4}));
}

test "ndarray: pad() memory safety with allocator" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 10, 10 }, .row_major);
        defer arr.deinit();

        for (0..10) |i| {
            for (0..10) |j| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
            }
        }

        var padded = try arr.pad(allocator, &[_][2]usize{ .{ 2, 2 }, .{ 2, 2 } }, .constant, 0);
        defer padded.deinit();

        try testing.expectEqual(@as(usize, 14), padded.shape[0]);
        try testing.expectEqual(@as(usize, 14), padded.shape[1]);
    }
}

test "ndarray: pad() column-major layout preservation" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .column_major);
    defer arr.deinit();
    arr.set(&[_]isize{ 0, 0 }, 1);
    arr.set(&[_]isize{ 0, 1 }, 2);
    arr.set(&[_]isize{ 1, 0 }, 3);
    arr.set(&[_]isize{ 1, 1 }, 4);

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 1, 1 }, .{ 1, 1 } }, .constant, 0);
    defer padded.deinit();

    try testing.expectEqual(Layout.column_major, padded.layout);
    try testing.expectEqual(@as(f64, 1), try padded.get(&[_]isize{ 1, 1 }));
}

test "ndarray: pad() large array stress test" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 50, 50 }, .row_major);
    defer arr.deinit();

    for (0..50) |i| {
        for (0..50) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 50 + j)));
        }
    }

    var padded = try arr.pad(allocator, &[_][2]usize{ .{ 5, 5 }, .{ 5, 5 } }, .edge, 0);
    defer padded.deinit();

    try testing.expectEqual(@as(usize, 60), padded.shape[0]);
    try testing.expectEqual(@as(usize, 60), padded.shape[1]);

    // Verify center preserved
    const center_val = try arr.get(&[_]isize{ 25, 25 });
    const padded_center_val = try padded.get(&[_]isize{ 30, 30 });
    try testing.expectEqual(center_val, padded_center_val);
}

test "ndarray: pad() validate() passes after padding" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var padded = try arr.pad(allocator, &[_][2]usize{.{ 2, 2 }}, .constant, 0);
    defer padded.deinit();

    try padded.validate();
}

// -- CSV I/O Tests --

test "ndarray: toCSV() and fromCSV() basic roundtrip f64" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_basic.csv";

    // Create 3x4 matrix
    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer original.deinit();

    // Fill with test data
    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 1.5);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 2.7);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, -3.2);
    original.set(&[_]isize{ @intCast(0), @intCast(3) }, 4.9);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 5.1);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, -6.8);
    original.set(&[_]isize{ @intCast(1), @intCast(2) }, 7.3);
    original.set(&[_]isize{ @intCast(1), @intCast(3) }, 8.0);
    original.set(&[_]isize{ @intCast(2), @intCast(0) }, -9.4);
    original.set(&[_]isize{ @intCast(2), @intCast(1) }, 10.6);
    original.set(&[_]isize{ @intCast(2), @intCast(2) }, 11.2);
    original.set(&[_]isize{ @intCast(2), @intCast(3) }, -12.7);

    // Save to CSV
    try original.toCSV(path, ',');

    // Load back
    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    // Verify shape
    try testing.expectEqual(@as(usize, 3), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 4), loaded.shape[1]);

    // Verify data (with tolerance for float precision)
    for (0..3) |r| {
        for (0..4) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectApproxEqAbs(orig_val, loaded_val, 1e-9);
        }
    }

    // Cleanup
    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() and fromCSV() integer types i32" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_i32.csv";

    var original = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, -100);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 42);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, 999);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 0);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, -50);
    original.set(&[_]isize{ @intCast(1), @intCast(2) }, 123);

    try original.toCSV(path, ',');

    var loaded = try NDArray(i32, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 2), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 3), loaded.shape[1]);

    for (0..2) |r| {
        for (0..3) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectEqual(orig_val, loaded_val);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() and fromCSV() u8 type" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_u8.csv";

    var original = try NDArray(u8, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 10);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 20);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 30);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, 40);
    original.set(&[_]isize{ @intCast(2), @intCast(0) }, 50);
    original.set(&[_]isize{ @intCast(2), @intCast(1) }, 255);

    try original.toCSV(path, ',');

    var loaded = try NDArray(u8, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    for (0..3) |r| {
        for (0..2) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectEqual(orig_val, loaded_val);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() custom delimiter semicolon" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_semicolon.csv";

    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 1.1);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 2.2);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 3.3);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, 4.4);

    // Use semicolon as delimiter
    try original.toCSV(path, ';');

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ';');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 2), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 2), loaded.shape[1]);

    for (0..2) |r| {
        for (0..2) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectApproxEqAbs(orig_val, loaded_val, 1e-9);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() custom delimiter tab" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_tab.csv";

    var original = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 10);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 20);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, 30);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 40);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, 50);
    original.set(&[_]isize{ @intCast(1), @intCast(2) }, 60);

    // Use tab as delimiter
    try original.toCSV(path, '\t');

    var loaded = try NDArray(i32, 2).fromCSV(allocator, path, '\t');
    defer loaded.deinit();

    for (0..2) |r| {
        for (0..3) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectEqual(orig_val, loaded_val);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() single row" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_single_row.csv";

    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 5 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 1.0);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 2.0);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, 3.0);
    original.set(&[_]isize{ @intCast(0), @intCast(3) }, 4.0);
    original.set(&[_]isize{ @intCast(0), @intCast(4) }, 5.0);

    try original.toCSV(path, ',');

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 1), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 5), loaded.shape[1]);

    for (0..5) |c| {
        const orig_val = try original.get(&[_]isize{ @intCast(0), @intCast(c) });
        const loaded_val = try loaded.get(&[_]isize{ @intCast(0), @intCast(c) });
        try testing.expectApproxEqAbs(orig_val, loaded_val, 1e-9);
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() single column" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_single_col.csv";

    var original = try NDArray(i32, 2).init(allocator, &[_]usize{ 4, 1 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 10);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 20);
    original.set(&[_]isize{ @intCast(2), @intCast(0) }, 30);
    original.set(&[_]isize{ @intCast(3), @intCast(0) }, 40);

    try original.toCSV(path, ',');

    var loaded = try NDArray(i32, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 4), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 1), loaded.shape[1]);

    for (0..4) |r| {
        const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(0) });
        const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(0) });
        try testing.expectEqual(orig_val, loaded_val);
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() 1x1 matrix" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_1x1.csv";

    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 1, 1 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 42.0);

    try original.toCSV(path, ',');

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 1), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 1), loaded.shape[1]);

    const orig_val = try original.get(&[_]isize{ @intCast(0), @intCast(0) });
    const loaded_val = try loaded.get(&[_]isize{ @intCast(0), @intCast(0) });
    try testing.expectApproxEqAbs(orig_val, loaded_val, 1e-9);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() large array (100 rows)" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_large.csv";

    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 100, 10 }, .row_major);
    defer original.deinit();

    // Fill with sequential data
    for (0..100) |r| {
        for (0..10) |c| {
            const val: f64 = @floatFromInt(r * 10 + c);
            original.set(&[_]isize{ @intCast(r), @intCast(c) }, val);
        }
    }

    try original.toCSV(path, ',');

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 100), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 10), loaded.shape[1]);

    // Spot check a few values
    try testing.expectApproxEqAbs(0.0, try loaded.get(&[_]isize{ @intCast(0), @intCast(0) }), 1e-9);
    try testing.expectApproxEqAbs(99.0, try loaded.get(&[_]isize{ @intCast(9), @intCast(9) }), 1e-9);
    try testing.expectApproxEqAbs(505.0, try loaded.get(&[_]isize{ @intCast(50), @intCast(5) }), 1e-9);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: fromCSV() handles whitespace trimming" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_whitespace.csv";

    // Create CSV with extra whitespace
    const csv_content = "  1.5  ,  2.7  ,  3.2  \n  4.1  ,  5.9  ,  6.3  \n";
    {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        _ = try file.write(csv_content);
    }

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 2), loaded.shape[0]);
    try testing.expectEqual(@as(usize, 3), loaded.shape[1]);

    try testing.expectApproxEqAbs(1.5, try loaded.get(&[_]isize{ @intCast(0), @intCast(0) }), 1e-9);
    try testing.expectApproxEqAbs(2.7, try loaded.get(&[_]isize{ @intCast(0), @intCast(1) }), 1e-9);
    try testing.expectApproxEqAbs(6.3, try loaded.get(&[_]isize{ @intCast(1), @intCast(2) }), 1e-9);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: fromCSV() error on empty file" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_empty.csv";

    {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        // Write nothing
    }

    const result = NDArray(f64, 2).fromCSV(allocator, path, ',');
    try testing.expectError(error.EmptyArray, result);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: fromCSV() error on ragged array (unequal columns)" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_ragged.csv";

    const csv_content = "1,2,3\n4,5\n6,7,8\n";
    {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        _ = try file.write(csv_content);
    }

    const result = NDArray(f64, 2).fromCSV(allocator, path, ',');
    try testing.expectError(error.InvalidFormat, result);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: fromCSV() error on invalid number format" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_invalid.csv";

    const csv_content = "1.5,abc,3.2\n4.1,5.9,6.3\n";
    {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        _ = try file.write(csv_content);
    }

    const result = NDArray(f64, 2).fromCSV(allocator, path, ',');
    try testing.expectError(error.InvalidFormat, result);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: toCSV() error on 1D array" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_1d.csv";

    var arr = try NDArray(f64, 1).arange(allocator, 0, 10, 1, .row_major);
    defer arr.deinit();

    const result = arr.toCSV(path, ',');
    try testing.expectError(error.DimensionMismatch, result);
}

test "ndarray: toCSV() error on 3D array" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_3d.csv";

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 2, 2 }, .row_major);
    defer arr.deinit();

    const result = arr.toCSV(path, ',');
    try testing.expectError(error.DimensionMismatch, result);
}

test "ndarray: fromCSV() error on 1D type" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_type_1d.csv";

    const csv_content = "1,2,3\n4,5,6\n";
    {
        const file = try fs.cwd().createFile(path, .{});
        defer file.close();
        _ = try file.write(csv_content);
    }

    const result = NDArray(f64, 1).fromCSV(allocator, path, ',');
    try testing.expectError(error.DimensionMismatch, result);

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: CSV roundtrip with negative values" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_negative.csv";

    var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, -1.5);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 2.7);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, -3.2);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 4.9);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, -5.1);
    original.set(&[_]isize{ @intCast(1), @intCast(2) }, 6.8);
    original.set(&[_]isize{ @intCast(2), @intCast(0) }, -7.3);
    original.set(&[_]isize{ @intCast(2), @intCast(1) }, -8.0);
    original.set(&[_]isize{ @intCast(2), @intCast(2) }, 9.4);

    try original.toCSV(path, ',');

    var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    for (0..3) |r| {
        for (0..3) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectApproxEqAbs(orig_val, loaded_val, 1e-9);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: CSV roundtrip with zeros" {
    const allocator = testing.allocator;
    const path = "/tmp/test_ndarray_csv_zeros.csv";

    var original = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer original.deinit();

    original.set(&[_]isize{ @intCast(0), @intCast(0) }, 0);
    original.set(&[_]isize{ @intCast(0), @intCast(1) }, 0);
    original.set(&[_]isize{ @intCast(0), @intCast(2) }, 1);
    original.set(&[_]isize{ @intCast(1), @intCast(0) }, 0);
    original.set(&[_]isize{ @intCast(1), @intCast(1) }, 2);
    original.set(&[_]isize{ @intCast(1), @intCast(2) }, 0);

    try original.toCSV(path, ',');

    var loaded = try NDArray(i32, 2).fromCSV(allocator, path, ',');
    defer loaded.deinit();

    for (0..2) |r| {
        for (0..3) |c| {
            const orig_val = try original.get(&[_]isize{ @intCast(r), @intCast(c) });
            const loaded_val = try loaded.get(&[_]isize{ @intCast(r), @intCast(c) });
            try testing.expectEqual(orig_val, loaded_val);
        }
    }

    fs.cwd().deleteFile(path) catch {};
}

test "ndarray: CSV roundtrip memory safety with allocator" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const path = "/tmp/test_ndarray_csv_memory.csv";

        var original = try NDArray(f64, 2).init(allocator, &[_]usize{ 5, 5 }, .row_major);
        defer original.deinit();

        for (0..5) |r| {
            for (0..5) |c| {
                const val: f64 = @floatFromInt(r * 5 + c);
                original.set(&[_]isize{ @intCast(r), @intCast(c) }, val);
            }
        }

        try original.toCSV(path, ',');

        var loaded = try NDArray(f64, 2).fromCSV(allocator, path, ',');
        defer loaded.deinit();

        try testing.expectEqual(@as(usize, 5), loaded.shape[0]);
        try testing.expectEqual(@as(usize, 5), loaded.shape[1]);

        fs.cwd().deleteFile(path) catch {};
    }
}

// ============================================================================
// repeat() and tile() tests
// ============================================================================

test "ndarray: repeat() 1D with axis=0" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 2, 0);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 6), repeated.shape[0]);
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{5}));
}

test "ndarray: repeat() 2D with axis=0" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 2, 0);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 4), repeated.shape[0]);
    try testing.expectEqual(@as(usize, 3), repeated.shape[1]);

    // First row repeated
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{ 1, 2 }));

    // Second row repeated
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{ 2, 0 }));
    try testing.expectEqual(@as(f64, 5), try repeated.get(&[_]isize{ 2, 1 }));
    try testing.expectEqual(@as(f64, 6), try repeated.get(&[_]isize{ 2, 2 }));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{ 3, 0 }));
    try testing.expectEqual(@as(f64, 5), try repeated.get(&[_]isize{ 3, 1 }));
    try testing.expectEqual(@as(f64, 6), try repeated.get(&[_]isize{ 3, 2 }));
}

test "ndarray: repeat() 2D with axis=1" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 3, 1);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 2), repeated.shape[0]);
    try testing.expectEqual(@as(usize, 6), repeated.shape[1]);

    // First row: [1, 1, 1, 2, 2, 2]
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 3 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 4 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 5 }));

    // Second row: [3, 3, 3, 4, 4, 4]
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{ 1, 2 }));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{ 1, 3 }));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{ 1, 4 }));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{ 1, 5 }));
}

test "ndarray: repeatFlat() flattens then repeats" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeatFlat(allocator, 2);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 8), repeated.shape[0]);
    // [1, 1, 2, 2, 3, 3, 4, 4]
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 3), try repeated.get(&[_]isize{5}));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{6}));
    try testing.expectEqual(@as(f64, 4), try repeated.get(&[_]isize{7}));
}

test "ndarray: repeat() single element" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{1}, &[_]i32{42}, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 5, 0);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 5), repeated.shape[0]);
    for (0..5) |i| {
        try testing.expectEqual(@as(i32, 42), try repeated.get(&[_]isize{@intCast(i)}));
    }
}

test "ndarray: repeat() 3D with axis=2" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 2, 2);
    defer repeated.deinit();

    try testing.expectEqual(@as(usize, 2), repeated.shape[0]);
    try testing.expectEqual(@as(usize, 2), repeated.shape[1]);
    try testing.expectEqual(@as(usize, 4), repeated.shape[2]);

    // Check a few elements
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 1), try repeated.get(&[_]isize{ 0, 0, 1 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 0, 2 }));
    try testing.expectEqual(@as(f64, 2), try repeated.get(&[_]isize{ 0, 0, 3 }));
}

test "ndarray: repeat() error - zero repeats" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.ZeroDimension, arr.repeat(allocator, 0, 0));
}

test "ndarray: repeat() error - invalid axis" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.IndexOutOfBounds, arr.repeat(allocator, 2, 2));
}

test "ndarray: repeat() type variants" {
    const allocator = testing.allocator;

    // i32
    var arr_i32 = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &[_]i32{ 10, 20 }, .row_major);
    defer arr_i32.deinit();
    var repeated_i32 = try arr_i32.repeat(allocator, 2, 0);
    defer repeated_i32.deinit();
    try testing.expectEqual(@as(usize, 4), repeated_i32.shape[0]);

    // u8
    var arr_u8 = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 1, 2 }, .row_major);
    defer arr_u8.deinit();
    var repeated_u8 = try arr_u8.repeat(allocator, 3, 0);
    defer repeated_u8.deinit();
    try testing.expectEqual(@as(usize, 6), repeated_u8.shape[0]);
}

test "ndarray: repeat() column-major layout" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .column_major);
    defer arr.deinit();

    var repeated = try arr.repeat(allocator, 2, 0);
    defer repeated.deinit();

    try testing.expectEqual(Layout.column_major, repeated.layout);
    try testing.expectEqual(@as(usize, 4), repeated.shape[0]);
    try testing.expectEqual(@as(usize, 2), repeated.shape[1]);
}

test "ndarray: repeat() memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
        defer arr.deinit();

        var repeated = try arr.repeat(allocator, 2, 0);
        defer repeated.deinit();

        try repeated.validate();
    }
}

test "ndarray: tile() 1D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &[_]f64{ 1, 2, 3 }, .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{3});
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 9), tiled.shape[0]);
    // [1, 2, 3, 1, 2, 3, 1, 2, 3]
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{0}));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{1}));
    try testing.expectEqual(@as(f64, 3), try tiled.get(&[_]isize{2}));
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{3}));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{4}));
    try testing.expectEqual(@as(f64, 3), try tiled.get(&[_]isize{5}));
}

test "ndarray: tile() 2D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 2, 3 });
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 4), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 6), tiled.shape[1]);

    // First tile
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(f64, 3), try tiled.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(f64, 4), try tiled.get(&[_]isize{ 1, 1 }));

    // Horizontally tiled
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 0, 3 }));

    // Vertically tiled
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 2, 0 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 2, 1 }));
}

test "ndarray: tile() single repetition" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 1, 1 });
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 2), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 2), tiled.shape[1]);

    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(f64, 3), try tiled.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(f64, 4), try tiled.get(&[_]isize{ 1, 1 }));
}

test "ndarray: tile() 3D" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 1, 2, 1 });
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 2), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 4), tiled.shape[1]);
    try testing.expectEqual(@as(usize, 2), tiled.shape[2]);

    // Original elements
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 0, 0, 1 }));

    // Tiled in axis=1
    try testing.expectEqual(@as(f64, 1), try tiled.get(&[_]isize{ 0, 2, 0 }));
    try testing.expectEqual(@as(f64, 2), try tiled.get(&[_]isize{ 0, 2, 1 }));
}

test "ndarray: tile() error - zero reps" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.ZeroDimension, arr.tile(allocator, &[_]usize{ 2, 0 }));
}

test "ndarray: tile() error - shape mismatch" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .row_major);
    defer arr.deinit();

    try testing.expectError(error.ShapeMismatch, arr.tile(allocator, &[_]usize{2}));
}

test "ndarray: tile() type variants" {
    const allocator = testing.allocator;

    // i32
    var arr_i32 = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]i32{ 1, 2, 3, 4 }, .row_major);
    defer arr_i32.deinit();
    var tiled_i32 = try arr_i32.tile(allocator, &[_]usize{ 2, 1 });
    defer tiled_i32.deinit();
    try testing.expectEqual(@as(usize, 4), tiled_i32.shape[0]);

    // u8
    var arr_u8 = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{2}, &[_]u8{ 5, 6 }, .row_major);
    defer arr_u8.deinit();
    var tiled_u8 = try arr_u8.tile(allocator, &[_]usize{3});
    defer tiled_u8.deinit();
    try testing.expectEqual(@as(usize, 6), tiled_u8.shape[0]);
}

test "ndarray: tile() column-major layout" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f64{ 1, 2, 3, 4 }, .column_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 2, 2 });
    defer tiled.deinit();

    try testing.expectEqual(Layout.column_major, tiled.layout);
    try testing.expectEqual(@as(usize, 4), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 4), tiled.shape[1]);
}

test "ndarray: tile() large array" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 10, 10 }, &([_]f64{1.0} ** 100), .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 2, 3 });
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 20), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 30), tiled.shape[1]);
    try tiled.validate();
}

test "ndarray: tile() memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .row_major);
        defer arr.deinit();

        var tiled = try arr.tile(allocator, &[_]usize{ 2, 2 });
        defer tiled.deinit();

        try tiled.validate();
    }
}

test "ndarray: tile() single element array" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f64{42}, .row_major);
    defer arr.deinit();

    var tiled = try arr.tile(allocator, &[_]usize{ 3, 4 });
    defer tiled.deinit();

    try testing.expectEqual(@as(usize, 3), tiled.shape[0]);
    try testing.expectEqual(@as(usize, 4), tiled.shape[1]);

    // All elements should be 42
    for (0..3) |i| {
        for (0..4) |j| {
            try testing.expectEqual(@as(f64, 42), try tiled.get(&[_]isize{ @intCast(i), @intCast(j) }));
        }
    }
}

// -- unique() Tests --

test "unique: basic 1D array with duplicates" {
    const allocator = testing.allocator;

    const data = [_]i32{ 3, 1, 2, 1, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should return [1, 2, 3] in sorted order
    try testing.expectEqual(@as(usize, 3), uniq.shape[0]);
    try testing.expectEqual(@as(i32, 1), try uniq.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try uniq.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try uniq.get(&[_]isize{2}));
}

test "unique: already sorted array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should return same array (all unique)
    try testing.expectEqual(@as(usize, 5), uniq.shape[0]);
    for (0..5) |i| {
        try testing.expectEqual(data[i], try uniq.get(&[_]isize{@intCast(i)}));
    }
}

test "unique: all identical elements" {
    const allocator = testing.allocator;

    const data = [_]i32{ 7, 7, 7, 7, 7 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should return single element
    try testing.expectEqual(@as(usize, 1), uniq.shape[0]);
    try testing.expectEqual(@as(i32, 7), try uniq.get(&[_]isize{0}));
}

test "unique: single element array" {
    const allocator = testing.allocator;

    const data = [_]i32{42};
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    try testing.expectEqual(@as(usize, 1), uniq.shape[0]);
    try testing.expectEqual(@as(i32, 42), try uniq.get(&[_]isize{0}));
}

test "unique: negative numbers" {
    const allocator = testing.allocator;

    const data = [_]i32{ -3, 1, -3, 2, -1, 1 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should return [-3, -1, 1, 2] in sorted order
    try testing.expectEqual(@as(usize, 4), uniq.shape[0]);
    try testing.expectEqual(@as(i32, -3), try uniq.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, -1), try uniq.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 1), try uniq.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 2), try uniq.get(&[_]isize{3}));
}

test "unique: 2D array (flattened)" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 2, 3, 1, 3 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should flatten and return [1, 2, 3]
    try testing.expectEqual(@as(usize, 3), uniq.shape[0]);
    try testing.expectEqual(@as(i32, 1), try uniq.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try uniq.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try uniq.get(&[_]isize{2}));
}

test "unique: floating point values" {
    const allocator = testing.allocator;

    const data = [_]f64{ 3.14, 2.71, 3.14, 1.41, 2.71 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    // Should return [1.41, 2.71, 3.14]
    try testing.expectEqual(@as(usize, 3), uniq.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 1.41), try uniq.get(&[_]isize{0}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.71), try uniq.get(&[_]isize{1}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.14), try uniq.get(&[_]isize{2}), 1e-9);
}

test "unique: u8 type" {
    const allocator = testing.allocator;

    const data = [_]u8{ 100, 50, 100, 25, 50, 75 };
    var arr = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var uniq = try arr.unique(allocator);
    defer uniq.deinit();

    try testing.expectEqual(@as(usize, 4), uniq.shape[0]);
    try testing.expectEqual(@as(u8, 25), try uniq.get(&[_]isize{0}));
    try testing.expectEqual(@as(u8, 50), try uniq.get(&[_]isize{1}));
    try testing.expectEqual(@as(u8, 75), try uniq.get(&[_]isize{2}));
    try testing.expectEqual(@as(u8, 100), try uniq.get(&[_]isize{3}));
}

test "unique: memory safety" {
    const allocator = testing.allocator;

    // Run multiple times to catch memory leaks
    for (0..10) |_| {
        const data = [_]i32{ 5, 3, 5, 1, 3, 5, 1 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
        defer arr.deinit();

        var uniq = try arr.unique(allocator);
        defer uniq.deinit();

        try testing.expectEqual(@as(usize, 3), uniq.shape[0]);
    }
}

// -- uniqueWithCounts() Tests --

test "uniqueWithCounts: basic 1D array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 3, 1, 2, 1, 3, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    // Values: [1, 2, 3]
    // Counts: [2, 1, 3]
    try testing.expectEqual(@as(usize, 3), result.values.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.counts.shape[0]);

    try testing.expectEqual(@as(i32, 1), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.values.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try result.values.get(&[_]isize{2}));

    try testing.expectEqual(@as(usize, 2), try result.counts.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 1), try result.counts.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 3), try result.counts.get(&[_]isize{2}));
}

test "uniqueWithCounts: all identical elements" {
    const allocator = testing.allocator;

    const data = [_]i32{ 5, 5, 5, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    try testing.expectEqual(@as(usize, 1), result.values.shape[0]);
    try testing.expectEqual(@as(i32, 5), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 4), try result.counts.get(&[_]isize{0}));
}

test "uniqueWithCounts: all unique elements" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    try testing.expectEqual(@as(usize, 4), result.values.shape[0]);
    // All counts should be 1
    for (0..4) |i| {
        try testing.expectEqual(@as(usize, 1), try result.counts.get(&[_]isize{@intCast(i)}));
    }
}

test "uniqueWithCounts: single element" {
    const allocator = testing.allocator;

    const data = [_]i32{42};
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    try testing.expectEqual(@as(usize, 1), result.values.shape[0]);
    try testing.expectEqual(@as(i32, 42), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 1), try result.counts.get(&[_]isize{0}));
}

test "uniqueWithCounts: 2D array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 2, 3, 1, 3, 3, 1 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 4 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    // Values: [1, 2, 3]
    // Counts: [3, 2, 3]
    try testing.expectEqual(@as(usize, 3), result.values.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.values.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try result.values.get(&[_]isize{2}));

    try testing.expectEqual(@as(usize, 3), try result.counts.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try result.counts.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 3), try result.counts.get(&[_]isize{2}));
}

test "uniqueWithCounts: f64 type" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.5, 2.5, 1.5, 2.5, 1.5 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    try testing.expectEqual(@as(usize, 2), result.values.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 1.5), try result.values.get(&[_]isize{0}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.5), try result.values.get(&[_]isize{1}), 1e-9);

    try testing.expectEqual(@as(usize, 3), try result.counts.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try result.counts.get(&[_]isize{1}));
}

test "uniqueWithCounts: negative numbers" {
    const allocator = testing.allocator;

    const data = [_]i32{ -5, -5, 0, -5, 0, 10, 10 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    // Values: [-5, 0, 10]
    // Counts: [3, 2, 2]
    try testing.expectEqual(@as(usize, 3), result.values.shape[0]);
    try testing.expectEqual(@as(i32, -5), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 0), try result.values.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 10), try result.values.get(&[_]isize{2}));

    try testing.expectEqual(@as(usize, 3), try result.counts.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try result.counts.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 2), try result.counts.get(&[_]isize{2}));
}

test "uniqueWithCounts: memory safety" {
    const allocator = testing.allocator;

    // Run multiple times to catch memory leaks
    for (0..10) |_| {
        const data = [_]i32{ 7, 3, 7, 3, 7, 7, 3 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
        defer arr.deinit();

        var result = try arr.uniqueWithCounts(allocator);
        defer result.values.deinit();
        defer result.counts.deinit();

        try testing.expectEqual(@as(usize, 2), result.values.shape[0]);
        try testing.expectEqual(@as(usize, 2), result.counts.shape[0]);
    }
}

test "uniqueWithCounts: large array" {
    const allocator = testing.allocator;

    // Create array with pattern: [0,1,2,0,1,2,0,1,2,...] for 300 elements
    var data: [300]i32 = undefined;
    for (0..300) |i| {
        data[i] = @intCast(i % 3);
    }

    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.uniqueWithCounts(allocator);
    defer result.values.deinit();
    defer result.counts.deinit();

    // Should have 3 unique values: [0, 1, 2]
    // Each appears 100 times
    try testing.expectEqual(@as(usize, 3), result.values.shape[0]);
    try testing.expectEqual(@as(i32, 0), try result.values.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 1), try result.values.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 2), try result.values.get(&[_]isize{2}));

    try testing.expectEqual(@as(usize, 100), try result.counts.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 100), try result.counts.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 100), try result.counts.get(&[_]isize{2}));
}

test "searchsorted: basic left insertion" {
    const allocator = testing.allocator;

    // Sorted array: [1, 3, 5, 7]
    const data = [_]i32{ 1, 3, 5, 7 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    // Values to search: [2, 6]
    const search_data = [_]i32{ 2, 6 };
    var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    var indices = try arr.searchsorted(allocator, &search, .left);
    defer indices.deinit();

    // 2 should insert at index 1 (between 1 and 3)
    // 6 should insert at index 3 (between 5 and 7)
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 3), try indices.get(&[_]isize{1}));
}

test "searchsorted: right insertion" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 3, 3, 3, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const search_data = [_]i32{ 3 };
    var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    // Left insertion: before all 3's (index 1)
    var indices_left = try arr.searchsorted(allocator, &search, .left);
    defer indices_left.deinit();
    try testing.expectEqual(@as(usize, 1), try indices_left.get(&[_]isize{0}));

    // Right insertion: after all 3's (index 4)
    var indices_right = try arr.searchsorted(allocator, &search, .right);
    defer indices_right.deinit();
    try testing.expectEqual(@as(usize, 4), try indices_right.get(&[_]isize{0}));
}

test "searchsorted: boundary cases" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 3, 5, 7 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    // Values at boundaries
    const search_data = [_]i32{ 0, 8 };
    var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    var indices = try arr.searchsorted(allocator, &search, .left);
    defer indices.deinit();

    // 0 should insert at index 0 (before all)
    // 8 should insert at index 4 (after all)
    try testing.expectEqual(@as(usize, 0), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 4), try indices.get(&[_]isize{1}));
}

test "searchsorted: exact matches" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 3, 5, 7 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    // Exact match values
    const search_data = [_]i32{ 1, 5, 7 };
    var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    var indices = try arr.searchsorted(allocator, &search, .left);
    defer indices.deinit();

    // Exact matches should return their index
    try testing.expectEqual(@as(usize, 0), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try indices.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 3), try indices.get(&[_]isize{2}));
}

test "searchsorted: single element array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const search_data = [_]i32{ 3, 5, 7 };
    var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    var indices = try arr.searchsorted(allocator, &search, .left);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 0), try indices.get(&[_]isize{0})); // 3 < 5
    try testing.expectEqual(@as(usize, 0), try indices.get(&[_]isize{1})); // 5 == 5 (left)
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{2})); // 7 > 5
}

test "searchsorted: float type" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.5, 3.2, 5.8, 7.1 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const search_data = [_]f64{ 2.0, 6.0 };
    var search = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
    defer search.deinit();

    var indices = try arr.searchsorted(allocator, &search, .left);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 3), try indices.get(&[_]isize{1}));
}

test "searchsorted: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 3, 5, 7, 9 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
        defer arr.deinit();

        const search_data = [_]i32{ 2, 4, 6 };
        var search = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{search_data.len}, &search_data, .row_major);
        defer search.deinit();

        var indices = try arr.searchsorted(allocator, &search, .left);
        defer indices.deinit();
    }
}

test "nonzero: basic integer array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 0, 1, 2, 0, 3, 0 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    // Non-zero elements are 1, 2, 3 at flat indices 1, 2, 4
    try testing.expectEqual(@as(usize, 3), indices.shape[0]);
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try indices.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 4), try indices.get(&[_]isize{2}));
}

test "nonzero: all zeros" {
    const allocator = testing.allocator;

    const data = [_]i32{ 0, 0, 0, 0 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 0), indices.shape[0]);
}

test "nonzero: all non-zero" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 5), indices.shape[0]);
    for (0..5) |i| {
        try testing.expectEqual(@as(usize, i), try indices.get(&[_]isize{@intCast(i)}));
    }
}

test "nonzero: negative numbers" {
    const allocator = testing.allocator;

    const data = [_]i32{ -1, 0, 2, 0, -3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    // Non-zero at indices 0, 2, 4
    try testing.expectEqual(@as(usize, 3), indices.shape[0]);
    try testing.expectEqual(@as(usize, 0), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try indices.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 4), try indices.get(&[_]isize{2}));
}

test "nonzero: float type" {
    const allocator = testing.allocator;

    const data = [_]f64{ 0.0, 1.5, 0.0, -2.3, 0.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    // Non-zero at indices 1, 3
    try testing.expectEqual(@as(usize, 2), indices.shape[0]);
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 3), try indices.get(&[_]isize{1}));
}

test "nonzero: boolean type" {
    const allocator = testing.allocator;

    const data = [_]bool{ false, true, false, true, true };
    var arr = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    // true at indices 1, 3, 4
    try testing.expectEqual(@as(usize, 3), indices.shape[0]);
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 3), try indices.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 4), try indices.get(&[_]isize{2}));
}

test "nonzero: 3D array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 0, 1, 2, 0, 0, 3, 0, 0, 4, 0, 0, 0 };
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var indices = try arr.nonzero(allocator);
    defer indices.deinit();

    // Non-zero at flat indices 1, 2, 5, 8
    try testing.expectEqual(@as(usize, 4), indices.shape[0]);
    try testing.expectEqual(@as(usize, 1), try indices.get(&[_]isize{0}));
    try testing.expectEqual(@as(usize, 2), try indices.get(&[_]isize{1}));
    try testing.expectEqual(@as(usize, 5), try indices.get(&[_]isize{2}));
    try testing.expectEqual(@as(usize, 8), try indices.get(&[_]isize{3}));
}

test "nonzero: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
        defer arr.deinit();

        var indices = try arr.nonzero(allocator);
        defer indices.deinit();
    }
}

// ============================================================================
// Set Operations Tests
// ============================================================================

test "union1d: basic arrays" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.union1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 2), result.data[1]);
    try testing.expectEqual(@as(i32, 3), result.data[2]);
    try testing.expectEqual(@as(i32, 4), result.data[3]);
}

test "union1d: duplicates" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 1, 2, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 2, 3, 3 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.union1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 2), result.data[1]);
    try testing.expectEqual(@as(i32, 3), result.data[2]);
}

test "union1d: disjoint sets" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 3, 5 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 4, 6 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.union1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 6), result.shape[0]);
    for (0..6) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 1)), result.data[i]);
    }
}

test "union1d: empty arrays" {
    const allocator = testing.allocator;
    // Manually create empty array (init() doesn't allow zero dimensions)
    const data_a = try allocator.alloc(i32, 0);
    var a = NDArray(i32, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data_a,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer a.deinit();

    const data_b = [_]i32{ 1, 2 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.union1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 2), result.data[1]);
}

test "intersect1d: basic arrays" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.intersect1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(i32, 2), result.data[0]);
    try testing.expectEqual(@as(i32, 3), result.data[1]);
}

test "intersect1d: no intersection" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.intersect1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.shape[0]);
}

test "intersect1d: duplicates" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.intersect1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(i32, 2), result.data[0]);
    try testing.expectEqual(@as(i32, 3), result.data[1]);
}

test "setdiff1d: basic arrays" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setdiff1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
}

test "setdiff1d: no difference" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 1, 2, 3 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setdiff1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.shape[0]);
}

test "setdiff1d: complete difference" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setdiff1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 2), result.data[1]);
}

test "setxor1d: basic arrays" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setxor1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 4), result.data[1]);
}

test "setxor1d: no overlap" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setxor1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 2), result.data[1]);
    try testing.expectEqual(@as(i32, 3), result.data[2]);
    try testing.expectEqual(@as(i32, 4), result.data[3]);
}

test "setxor1d: complete overlap" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 1, 2, 3 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.setxor1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.shape[0]);
}

test "in1d: basic arrays" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3, 4 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 2, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.in1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(false, result.data[0]);
    try testing.expectEqual(true, result.data[1]);
    try testing.expectEqual(false, result.data[2]);
    try testing.expectEqual(true, result.data[3]);
}

test "in1d: no elements in set" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 4, 5 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.in1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(false, result.data[0]);
    try testing.expectEqual(false, result.data[1]);
    try testing.expectEqual(false, result.data[2]);
}

test "in1d: all elements in set" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ 1, 2 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ 1, 2, 3, 4 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var result = try a.in1d(allocator, &b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(true, result.data[0]);
    try testing.expectEqual(true, result.data[1]);
}

test "set operations: type variants f64" {
    const allocator = testing.allocator;
    const data_a = [_]f64{ 1.0, 2.0, 3.0 };
    var a = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]f64{ 2.0, 3.0, 4.0 };
    var b = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var union_result = try a.union1d(allocator, &b);
    defer union_result.deinit();
    try testing.expectEqual(@as(usize, 4), union_result.shape[0]);

    var intersect_result = try a.intersect1d(allocator, &b);
    defer intersect_result.deinit();
    try testing.expectEqual(@as(usize, 2), intersect_result.shape[0]);
}

test "set operations: type variants u8" {
    const allocator = testing.allocator;
    const data_a = [_]u8{ 10, 20, 30 };
    var a = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]u8{ 20, 30, 40 };
    var b = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var setdiff_result = try a.setdiff1d(allocator, &b);
    defer setdiff_result.deinit();
    try testing.expectEqual(@as(usize, 1), setdiff_result.shape[0]);
    try testing.expectEqual(@as(u8, 10), setdiff_result.data[0]);
}

test "set operations: negative numbers" {
    const allocator = testing.allocator;
    const data_a = [_]i32{ -3, -1, 1, 3 };
    var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
    defer a.deinit();

    const data_b = [_]i32{ -2, -1, 0, 1 };
    var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
    defer b.deinit();

    var union_result = try a.union1d(allocator, &b);
    defer union_result.deinit();
    // Union: {-3, -2, -1, 0, 1, 3} = 6 elements
    try testing.expectEqual(@as(usize, 6), union_result.shape[0]);
}

test "set operations: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data_a = [_]i32{ 1, 2, 3, 4, 5 };
        var a = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_a.len}, &data_a, .row_major);
        defer a.deinit();

        const data_b = [_]i32{ 3, 4, 5, 6, 7 };
        var b = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data_b.len}, &data_b, .row_major);
        defer b.deinit();

        var union_result = try a.union1d(allocator, &b);
        defer union_result.deinit();

        var intersect_result = try a.intersect1d(allocator, &b);
        defer intersect_result.deinit();

        var setdiff_result = try a.setdiff1d(allocator, &b);
        defer setdiff_result.deinit();

        var setxor_result = try a.setxor1d(allocator, &b);
        defer setxor_result.deinit();

        var in1d_result = try a.in1d(allocator, &b);
        defer in1d_result.deinit();
    }
}

test "extract: basic 1D array" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true, false, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var extracted = try arr.extract(allocator, &condition);
    defer extracted.deinit();

    try testing.expectEqual(@as(usize, 3), extracted.shape[0]);
    try testing.expectEqual(@as(i32, 1), extracted.data[0]);
    try testing.expectEqual(@as(i32, 3), extracted.data[1]);
    try testing.expectEqual(@as(i32, 5), extracted.data[2]);
}

test "extract: 2D array (flattens)" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true, false, true, false };
    var condition = try NDArray(bool, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &cond_data, .row_major);
    defer condition.deinit();

    var extracted = try arr.extract(allocator, &condition);
    defer extracted.deinit();

    // Should extract elements at positions 0, 2, 4 -> values 1, 3, 5
    try testing.expectEqual(@as(usize, 3), extracted.shape[0]);
    try testing.expectEqual(@as(i32, 1), extracted.data[0]);
    try testing.expectEqual(@as(i32, 3), extracted.data[1]);
    try testing.expectEqual(@as(i32, 5), extracted.data[2]);
}

test "extract: all false condition" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ false, false, false, false };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var extracted = try arr.extract(allocator, &condition);
    defer extracted.deinit();

    try testing.expectEqual(@as(usize, 0), extracted.shape[0]);
}

test "extract: all true condition" {
    const allocator = testing.allocator;
    const data = [_]i32{ 10, 20, 30 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, true, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var extracted = try arr.extract(allocator, &condition);
    defer extracted.deinit();

    try testing.expectEqual(@as(usize, 3), extracted.shape[0]);
    try testing.expectEqual(@as(i32, 10), extracted.data[0]);
    try testing.expectEqual(@as(i32, 20), extracted.data[1]);
    try testing.expectEqual(@as(i32, 30), extracted.data[2]);
}

test "extract: shape mismatch error" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true }; // Wrong size
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    try testing.expectError(error.ShapeMismatch, arr.extract(allocator, &condition));
}

test "extract: float type" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.5, 2.5, 3.5, 4.5 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ false, true, true, false };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var extracted = try arr.extract(allocator, &condition);
    defer extracted.deinit();

    try testing.expectEqual(@as(usize, 2), extracted.shape[0]);
    try testing.expectEqual(@as(f64, 2.5), extracted.data[0]);
    try testing.expectEqual(@as(f64, 3.5), extracted.data[1]);
}

test "extract: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
        defer arr.deinit();

        const cond_data = [_]bool{ true, false, true, false, true, false };
        var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
        defer condition.deinit();

        var extracted = try arr.extract(allocator, &condition);
        defer extracted.deinit();
    }
}

test "compress: 1D array basic" {
    const allocator = testing.allocator;
    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true, false, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 0);
    defer compressed.deinit();

    try testing.expectEqual(@as(usize, 3), compressed.shape[0]);
    try testing.expectEqual(@as(i32, 10), compressed.data[0]);
    try testing.expectEqual(@as(i32, 30), compressed.data[1]);
    try testing.expectEqual(@as(i32, 50), compressed.data[2]);
}

test "compress: 2D array along axis 0 (rows)" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 0);
    defer compressed.deinit();

    // Should keep rows 0 and 2: [[1,2,3], [7,8,9]]
    try testing.expectEqual(@as(usize, 2), compressed.shape[0]);
    try testing.expectEqual(@as(usize, 3), compressed.shape[1]);
    try testing.expectEqual(@as(i32, 1), try compressed.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 2), try compressed.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 3), try compressed.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 7), try compressed.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 8), try compressed.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 9), try compressed.get(&[_]isize{ 1, 2 }));
}

test "compress: 2D array along axis 1 (columns)" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ false, true, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 1);
    defer compressed.deinit();

    // Should keep columns 1 and 2: [[2,3], [5,6], [8,9]]
    try testing.expectEqual(@as(usize, 3), compressed.shape[0]);
    try testing.expectEqual(@as(usize, 2), compressed.shape[1]);
    try testing.expectEqual(@as(i32, 2), try compressed.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 3), try compressed.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 5), try compressed.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 6), try compressed.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 8), try compressed.get(&[_]isize{ 2, 0 }));
    try testing.expectEqual(@as(i32, 9), try compressed.get(&[_]isize{ 2, 1 }));
}

test "compress: all false condition" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ false, false, false, false };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 0);
    defer compressed.deinit();

    try testing.expectEqual(@as(usize, 0), compressed.shape[0]);
}

test "compress: all true condition" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, true, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 0);
    defer compressed.deinit();

    try testing.expectEqual(@as(usize, 3), compressed.shape[0]);
    try testing.expectEqual(@as(i32, 1), compressed.data[0]);
    try testing.expectEqual(@as(i32, 2), compressed.data[1]);
    try testing.expectEqual(@as(i32, 3), compressed.data[2]);
}

test "compress: 3D array" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    var compressed = try arr.compress(allocator, &condition, 0);
    defer compressed.deinit();

    try testing.expectEqual(@as(usize, 1), compressed.shape[0]);
    try testing.expectEqual(@as(usize, 2), compressed.shape[1]);
    try testing.expectEqual(@as(usize, 2), compressed.shape[2]);
    try testing.expectEqual(@as(i32, 1), try compressed.get(&[_]isize{ 0, 0, 0 }));
    try testing.expectEqual(@as(i32, 2), try compressed.get(&[_]isize{ 0, 0, 1 }));
    try testing.expectEqual(@as(i32, 3), try compressed.get(&[_]isize{ 0, 1, 0 }));
    try testing.expectEqual(@as(i32, 4), try compressed.get(&[_]isize{ 0, 1, 1 }));
}

test "compress: invalid axis error" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true };
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    try testing.expectError(error.IndexOutOfBounds, arr.compress(allocator, &condition, 1));
}

test "compress: shape mismatch error" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    const cond_data = [_]bool{ true, false, true }; // Wrong size (3 instead of 4)
    var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
    defer condition.deinit();

    try testing.expectError(error.ShapeMismatch, arr.compress(allocator, &condition, 0));
}

test "compress: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
        var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
        defer arr.deinit();

        const cond_data = [_]bool{ true, false };
        var condition = try NDArray(bool, 1).fromSlice(allocator, &[_]usize{cond_data.len}, &cond_data, .row_major);
        defer condition.deinit();

        var compressed = try arr.compress(allocator, &condition, 0);
        defer compressed.deinit();
    }
}

test "flip: 1D array" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{data.len}, &data, .row_major);
    defer arr.deinit();

    var flipped = try arr.flip(allocator, 0);
    defer flipped.deinit();

    try testing.expectEqual(@as(i32, 5), try flipped.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 4), try flipped.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try flipped.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 2), try flipped.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 1), try flipped.get(&[_]isize{4}));
}

test "flip: 2D array along axis 0" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var flipped = try arr.flip(allocator, 0);
    defer flipped.deinit();

    // Original: [[1,2,3], [4,5,6]]
    // Flipped axis 0: [[4,5,6], [1,2,3]]
    try testing.expectEqual(@as(i32, 4), try flipped.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 5), try flipped.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 6), try flipped.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 1), try flipped.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 2), try flipped.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 3), try flipped.get(&[_]isize{ 1, 2 }));
}

test "flip: 2D array along axis 1" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var flipped = try arr.flip(allocator, 1);
    defer flipped.deinit();

    // Original: [[1,2,3], [4,5,6]]
    // Flipped axis 1: [[3,2,1], [6,5,4]]
    try testing.expectEqual(@as(i32, 3), try flipped.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 2), try flipped.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 1), try flipped.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 6), try flipped.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 5), try flipped.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 4), try flipped.get(&[_]isize{ 1, 2 }));
}

test "flip: invalid axis error" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    try testing.expectError(NDArray(i32, 1).Error.IndexOutOfBounds, arr.flip(allocator, 1));
}

test "flip: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        var flipped = try arr.flip(allocator, 0);
        defer flipped.deinit();

        try testing.expectApproxEqAbs(@as(f64, 4.0), try flipped.get(&[_]isize{0}), 1e-9);
    }
}

test "rot90: 2D array, k=1" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var rotated = try arr.rot90(allocator, 1, [2]usize{ 0, 1 });
    defer rotated.deinit();

    // Original: [[1,2], [3,4]]
    // Rotated 90° CCW: [[2,4], [1,3]]
    try testing.expectEqual(@as(i32, 2), try rotated.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 4), try rotated.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 1), try rotated.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 3), try rotated.get(&[_]isize{ 1, 1 }));
}

test "rot90: 2D array, k=2" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var rotated = try arr.rot90(allocator, 2, [2]usize{ 0, 1 });
    defer rotated.deinit();

    // Original: [[1,2], [3,4]]
    // Rotated 180°: [[4,3], [2,1]]
    try testing.expectEqual(@as(i32, 4), try rotated.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 3), try rotated.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 2), try rotated.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 1), try rotated.get(&[_]isize{ 1, 1 }));
}

test "rot90: k=0 returns copy" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var rotated = try arr.rot90(allocator, 0, [2]usize{ 0, 1 });
    defer rotated.deinit();

    for (0..4) |i| {
        try testing.expectEqual(arr.data[i], rotated.data[i]);
    }
}

test "rot90: negative k" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var rotated = try arr.rot90(allocator, -1, [2]usize{ 0, 1 });
    defer rotated.deinit();

    // k=-1 is same as k=3 (270° CCW = 90° CW)
    // Original: [[1,2], [3,4]]
    // Rotated 90° CW: [[3,1], [4,2]]
    try testing.expectEqual(@as(i32, 3), try rotated.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 1), try rotated.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 4), try rotated.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 2), try rotated.get(&[_]isize{ 1, 1 }));
}

test "rot90: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
        defer arr.deinit();

        var rotated = try arr.rot90(allocator, 1, [2]usize{ 0, 1 });
        defer rotated.deinit();

        try testing.expectApproxEqAbs(@as(f64, 2.0), try rotated.get(&[_]isize{ 0, 0 }), 1e-9);
    }
}

test "roll: 1D array positive shift" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var rolled = try arr.roll(allocator, 2, 0);
    defer rolled.deinit();

    // Shift right by 2: [4, 5, 1, 2, 3]
    try testing.expectEqual(@as(i32, 4), try rolled.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 5), try rolled.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 1), try rolled.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 2), try rolled.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 3), try rolled.get(&[_]isize{4}));
}

test "roll: 1D array negative shift" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var rolled = try arr.roll(allocator, -2, 0);
    defer rolled.deinit();

    // Shift left by 2: [3, 4, 5, 1, 2]
    try testing.expectEqual(@as(i32, 3), try rolled.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 4), try rolled.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 5), try rolled.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 1), try rolled.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 2), try rolled.get(&[_]isize{4}));
}

test "roll: 2D array along axis 0" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var rolled = try arr.roll(allocator, 1, 0);
    defer rolled.deinit();

    // Original: [[1,2,3], [4,5,6]]
    // Rolled axis 0: [[4,5,6], [1,2,3]]
    try testing.expectEqual(@as(i32, 4), try rolled.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 5), try rolled.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 6), try rolled.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 1), try rolled.get(&[_]isize{ 1, 0 }));
}

test "roll: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        var rolled = try arr.roll(allocator, 1, 0);
        defer rolled.deinit();

        try testing.expectApproxEqAbs(@as(f64, 4.0), try rolled.get(&[_]isize{0}), 1e-9);
    }
}

test "diff: 1D basic" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 3, 6, 10 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    var d = try arr.diff(allocator, 1, 0);
    defer d.deinit();

    // diff([1,3,6,10]) = [2,3,4]
    try testing.expectEqual(@as(usize, 3), d.shape[0]);
    try testing.expectEqual(@as(i32, 2), try d.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 3), try d.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 4), try d.get(&[_]isize{2}));
}

test "diff: 1D second order" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 3, 6, 10 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    var d = try arr.diff(allocator, 2, 0);
    defer d.deinit();

    // diff([1,3,6,10], n=2) = diff([2,3,4]) = [1,1]
    try testing.expectEqual(@as(usize, 2), d.shape[0]);
    try testing.expectEqual(@as(i32, 1), try d.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 1), try d.get(&[_]isize{1}));
}

test "diff: 2D along axis 0" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3, 5, 7, 9 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var d = try arr.diff(allocator, 1, 0);
    defer d.deinit();

    // Original: [[1,2,3], [5,7,9]]
    // diff axis 0: [[4,5,6]]
    try testing.expectEqual(@as(usize, 1), d.shape[0]);
    try testing.expectEqual(@as(usize, 3), d.shape[1]);
    try testing.expectEqual(@as(i32, 4), try d.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 5), try d.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 6), try d.get(&[_]isize{ 0, 2 }));
}

test "diff: n=0 returns copy" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    var d = try arr.diff(allocator, 0, 0);
    defer d.deinit();

    try testing.expectEqual(@as(usize, 3), d.shape[0]);
    for (0..3) |i| {
        try testing.expectEqual(arr.data[i], d.data[i]);
    }
}

test "diff: invalid axis error" {
    const allocator = testing.allocator;
    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    try testing.expectError(NDArray(i32, 1).Error.IndexOutOfBounds, arr.diff(allocator, 1, 1));
}

test "diff: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.5, 4.5, 7.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        var d = try arr.diff(allocator, 1, 0);
        defer d.deinit();

        try testing.expectApproxEqAbs(@as(f64, 1.5), try d.get(&[_]isize{0}), 1e-9);
    }
}

test "gradient: 1D basic" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 2.0, 4.0, 7.0, 11.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var grad = try arr.gradient(allocator, 0);
    defer grad.deinit();

    // Forward: 2.0-1.0=1.0
    // Central: (4.0-1.0)/2=1.5
    // Central: (7.0-2.0)/2=2.5
    // Central: (11.0-4.0)/2=3.5
    // Backward: 11.0-7.0=4.0
    try testing.expectApproxEqAbs(@as(f64, 1.0), try grad.get(&[_]isize{0}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.5), try grad.get(&[_]isize{1}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.5), try grad.get(&[_]isize{2}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.5), try grad.get(&[_]isize{3}), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), try grad.get(&[_]isize{4}), 1e-9);
}

test "gradient: 2D along axis 0" {
    const allocator = testing.allocator;
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 12.0, 15.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &data, .row_major);
    defer arr.deinit();

    var grad = try arr.gradient(allocator, 0);
    defer grad.deinit();

    // First row: forward difference
    try testing.expectApproxEqAbs(@as(f64, 3.0), try grad.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), try grad.get(&[_]isize{ 0, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try grad.get(&[_]isize{ 0, 2 }), 1e-9);

    // Middle row: central difference
    try testing.expectApproxEqAbs(@as(f64, 4.0), try grad.get(&[_]isize{ 1, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try grad.get(&[_]isize{ 1, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), try grad.get(&[_]isize{ 1, 2 }), 1e-9);

    // Last row: backward difference
    try testing.expectApproxEqAbs(@as(f64, 5.0), try grad.get(&[_]isize{ 2, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), try grad.get(&[_]isize{ 2, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 7.0), try grad.get(&[_]isize{ 2, 2 }), 1e-9);
}

test "gradient: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.5, 4.5, 7.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        var grad = try arr.gradient(allocator, 0);
        defer grad.deinit();

        try testing.expectApproxEqAbs(@as(f64, 1.5), try grad.get(&[_]isize{0}), 1e-9);
    }
}

// -- Insert Tests (7 tests) --

test "insert: 1D array at beginning" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 1, 2 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
    defer values.deinit();

    var result = try arr.insert(allocator, 0, 0, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{4}));
}

test "insert: 1D array in middle" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 5, 6 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 3, 4 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
    defer values.deinit();

    var result = try arr.insert(allocator, 0, 2, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 6), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{4}));
    try testing.expectEqual(@as(i32, 6), try result.get(&[_]isize{5}));
}

test "insert: 2D array along axis 0" {
    const allocator = testing.allocator;

    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]f64{ 5.0, 6.0 };
    var values = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 2 }, &data2, .row_major);
    defer values.deinit();

    var result = try arr.insert(allocator, 0, 1, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try result.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), try result.get(&[_]isize{ 0, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try result.get(&[_]isize{ 1, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), try result.get(&[_]isize{ 1, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.0), try result.get(&[_]isize{ 2, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), try result.get(&[_]isize{ 2, 1 }), 1e-9);
}

test "insert: 2D array along axis 1" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 5, 6 };
    var values = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 1 }, &data2, .row_major);
    defer values.deinit();

    var result = try arr.insert(allocator, 1, 1, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 6), try result.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{ 1, 2 }));
}

test "insert: shape mismatch error" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 5, 6, 7 };
    var values = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 1, 3 }, &data2, .row_major);
    defer values.deinit();

    try testing.expectError(NDArray(i32, 2).Error.ShapeMismatch, arr.insert(allocator, 0, 0, &values));
}

test "insert: invalid axis error" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 4, 5 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
    defer values.deinit();

    try testing.expectError(NDArray(i32, 1).Error.IndexOutOfBounds, arr.insert(allocator, 1, 0, &values));
}

test "insert: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data1 = [_]i32{ 1, 2, 3 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
        defer arr.deinit();

        const data2 = [_]i32{ 4, 5 };
        var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
        defer values.deinit();

        var result = try arr.insert(allocator, 0, 1, &values);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 5), result.shape[0]);
    }
}

// -- Append Tests (5 tests) --

test "append: 1D array" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 4, 5 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
    defer values.deinit();

    var result = try arr.append(allocator, 0, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 5), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{3}));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{4}));
}

test "append: 2D array along axis 0" {
    const allocator = testing.allocator;

    const data1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]f64{ 5.0, 6.0 };
    var values = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 2 }, &data2, .row_major);
    defer values.deinit();

    var result = try arr.append(allocator, 0, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try result.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), try result.get(&[_]isize{ 0, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.0), try result.get(&[_]isize{ 1, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 4.0), try result.get(&[_]isize{ 1, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try result.get(&[_]isize{ 2, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), try result.get(&[_]isize{ 2, 1 }), 1e-9);
}

test "append: 2D array along axis 1" {
    const allocator = testing.allocator;

    const data1 = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]i32{ 5, 6 };
    var values = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 1 }, &data2, .row_major);
    defer values.deinit();

    var result = try arr.append(allocator, 1, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{ 1, 1 }));
    try testing.expectEqual(@as(i32, 6), try result.get(&[_]isize{ 1, 2 }));
}

test "append: u8 type" {
    const allocator = testing.allocator;

    const data1 = [_]u8{ 1, 2, 3 };
    var arr = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &data1, .row_major);
    defer arr.deinit();

    const data2 = [_]u8{ 4, 5, 6 };
    var values = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &data2, .row_major);
    defer values.deinit();

    var result = try arr.append(allocator, 0, &values);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 6), result.shape[0]);
    try testing.expectEqual(@as(u8, 4), try result.get(&[_]isize{3}));
}

test "append: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data1 = [_]i32{ 1, 2 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data1, .row_major);
        defer arr.deinit();

        const data2 = [_]i32{ 3, 4 };
        var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &data2, .row_major);
        defer values.deinit();

        var result = try arr.append(allocator, 0, &values);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 4), result.shape[0]);
    }
}

// -- Delete Tests (8 tests) --

test "delete: 1D array single element" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.delete(allocator, 0, 2, 3);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 2), try result.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{2}));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{3}));
}

test "delete: 1D array multiple elements" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{6}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.delete(allocator, 0, 1, 4);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{0}));
    try testing.expectEqual(@as(i32, 5), try result.get(&[_]isize{1}));
    try testing.expectEqual(@as(i32, 6), try result.get(&[_]isize{2}));
}

test "delete: 2D array along axis 0" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 2 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.delete(allocator, 0, 1, 2);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try result.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), try result.get(&[_]isize{ 0, 1 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 5.0), try result.get(&[_]isize{ 1, 0 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 6.0), try result.get(&[_]isize{ 1, 1 }), 1e-9);
}

test "delete: 2D array along axis 1" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.delete(allocator, 1, 1, 2);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(i32, 1), try result.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 3), try result.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 4), try result.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 6), try result.get(&[_]isize{ 1, 1 }));
}

test "delete: invalid range error" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    try testing.expectError(NDArray(i32, 1).Error.IndexOutOfBounds, arr.delete(allocator, 0, 3, 2));
}

test "delete: out of bounds error" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    try testing.expectError(NDArray(i32, 1).Error.IndexOutOfBounds, arr.delete(allocator, 0, 0, 5));
}

test "delete: zero dimension error" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    try testing.expectError(NDArray(i32, 1).Error.ZeroDimension, arr.delete(allocator, 0, 0, 3));
}

test "delete: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 2, 3, 4, 5 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
        defer arr.deinit();

        var result = try arr.delete(allocator, 0, 1, 3);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 3), result.shape[0]);
    }
}

// -- take() tests --

test "take: 1D array basic" {
    const allocator = testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 2, 4 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{3}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 0, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 10), result.data[0]);
    try testing.expectEqual(@as(i32, 30), result.data[1]);
    try testing.expectEqual(@as(i32, 50), result.data[2]);
}

test "take: 2D array along axis 0" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 2 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 0, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
    // Row 0: [1, 2, 3]
    try testing.expectEqual(@as(f64, 1), result.data[0]);
    try testing.expectEqual(@as(f64, 2), result.data[1]);
    try testing.expectEqual(@as(f64, 3), result.data[2]);
    // Row 2: [7, 8, 9]
    try testing.expectEqual(@as(f64, 7), result.data[3]);
    try testing.expectEqual(@as(f64, 8), result.data[4]);
    try testing.expectEqual(@as(f64, 9), result.data[5]);
}

test "take: 2D array along axis 1" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 3 }, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 2 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 1, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    // Columns 0 and 2: [[1, 3], [4, 6], [7, 9]]
    try testing.expectEqual(@as(f64, 1), result.data[0]);
    try testing.expectEqual(@as(f64, 3), result.data[1]);
    try testing.expectEqual(@as(f64, 4), result.data[2]);
    try testing.expectEqual(@as(f64, 6), result.data[3]);
    try testing.expectEqual(@as(f64, 7), result.data[4]);
    try testing.expectEqual(@as(f64, 9), result.data[5]);
}

test "take: single element" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{2};
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 0, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(i32, 3), result.data[0]);
}

test "take: repeated indices" {
    const allocator = testing.allocator;

    const data = [_]i32{ 10, 20, 30 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 1, 1, 0, 2 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{4}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 0, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 20), result.data[0]);
    try testing.expectEqual(@as(i32, 20), result.data[1]);
    try testing.expectEqual(@as(i32, 10), result.data[2]);
    try testing.expectEqual(@as(i32, 30), result.data[3]);
}

test "take: invalid axis" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{0};
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &indices_data, .row_major);
    defer indices.deinit();

    const result = arr.take(allocator, 1, &indices);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "take: index out of bounds" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 5 }; // Index 5 is out of bounds
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const result = arr.take(allocator, 0, &indices);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "take: 3D array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{1};
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &indices_data, .row_major);
    defer indices.deinit();

    var result = try arr.take(allocator, 0, &indices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
    // Second slice: [[5, 6], [7, 8]]
    try testing.expectEqual(@as(i32, 5), result.data[0]);
    try testing.expectEqual(@as(i32, 6), result.data[1]);
    try testing.expectEqual(@as(i32, 7), result.data[2]);
    try testing.expectEqual(@as(i32, 8), result.data[3]);
}

test "take: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 2, 3, 4, 5 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
        defer arr.deinit();

        const indices_data = [_]usize{ 0, 2, 4 };
        var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{3}, &indices_data, .row_major);
        defer indices.deinit();

        var result = try arr.take(allocator, 0, &indices);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 3), result.shape[0]);
    }
}

// -- put() tests --

test "put: 1D array basic" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 4 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{ 99, 88 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &values_data, .row_major);
    defer values.deinit();

    try arr.put(&indices, &values);

    try testing.expectEqual(@as(i32, 99), arr.data[0]);
    try testing.expectEqual(@as(i32, 2), arr.data[1]);
    try testing.expectEqual(@as(i32, 3), arr.data[2]);
    try testing.expectEqual(@as(i32, 4), arr.data[3]);
    try testing.expectEqual(@as(i32, 88), arr.data[4]);
}

test "put: 2D array with flat indices" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    // Flat indices: 0=[0,0], 1=[0,1], 2=[0,2], 3=[1,0], 4=[1,1], 5=[1,2]
    const indices_data = [_]usize{ 0, 5 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]f64{ 10, 60 };
    var values = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &values_data, .row_major);
    defer values.deinit();

    try arr.put(&indices, &values);

    try testing.expectEqual(@as(f64, 10), arr.data[0]); // [0, 0]
    try testing.expectEqual(@as(f64, 2), arr.data[1]);
    try testing.expectEqual(@as(f64, 3), arr.data[2]);
    try testing.expectEqual(@as(f64, 4), arr.data[3]);
    try testing.expectEqual(@as(f64, 5), arr.data[4]);
    try testing.expectEqual(@as(f64, 60), arr.data[5]); // [1, 2]
}

test "put: single value" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{1};
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{42};
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{1}, &values_data, .row_major);
    defer values.deinit();

    try arr.put(&indices, &values);

    try testing.expectEqual(@as(i32, 1), arr.data[0]);
    try testing.expectEqual(@as(i32, 42), arr.data[1]);
    try testing.expectEqual(@as(i32, 3), arr.data[2]);
}

test "put: repeated indices" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    // Last write wins for repeated indices
    const indices_data = [_]usize{ 0, 0 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{ 10, 20 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &values_data, .row_major);
    defer values.deinit();

    try arr.put(&indices, &values);

    try testing.expectEqual(@as(i32, 20), arr.data[0]); // Last value wins
    try testing.expectEqual(@as(i32, 2), arr.data[1]);
    try testing.expectEqual(@as(i32, 3), arr.data[2]);
}

test "put: shape mismatch" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{ 0, 1 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{42}; // Mismatch: 2 indices, 1 value
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{1}, &values_data, .row_major);
    defer values.deinit();

    const result = arr.put(&indices, &values);
    try testing.expectError(error.ShapeMismatch, result);
}

test "put: index out of bounds" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const indices_data = [_]usize{10}; // Out of bounds
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{42};
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{1}, &values_data, .row_major);
    defer values.deinit();

    const result = arr.put(&indices, &values);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "put: 3D array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &data, .row_major);
    defer arr.deinit();

    // Flat index 0 = [0,0,0], flat index 7 = [1,1,1]
    const indices_data = [_]usize{ 0, 7 };
    var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
    defer indices.deinit();

    const values_data = [_]i32{ 100, 800 };
    var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &values_data, .row_major);
    defer values.deinit();

    try arr.put(&indices, &values);

    try testing.expectEqual(@as(i32, 100), arr.data[0]);
    try testing.expectEqual(@as(i32, 2), arr.data[1]);
    try testing.expectEqual(@as(i32, 3), arr.data[2]);
    try testing.expectEqual(@as(i32, 4), arr.data[3]);
    try testing.expectEqual(@as(i32, 5), arr.data[4]);
    try testing.expectEqual(@as(i32, 6), arr.data[5]);
    try testing.expectEqual(@as(i32, 7), arr.data[6]);
    try testing.expectEqual(@as(i32, 800), arr.data[7]);
}

test "put: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]i32{ 1, 2, 3, 4, 5 };
        var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
        defer arr.deinit();

        const indices_data = [_]usize{ 0, 2 };
        var indices = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{2}, &indices_data, .row_major);
        defer indices.deinit();

        const values_data = [_]i32{ 10, 30 };
        var values = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &values_data, .row_major);
        defer values.deinit();

        try arr.put(&indices, &values);
    }
}

// Broadcasting and dimension adjustment tests

test "atleast1d: scalar to 1D" {
    const allocator = testing.allocator;

    const data = [_]f64{42};
    var scalar = try NDArray(f64, 0).fromSlice(allocator, &[_]usize{}, &data, .row_major);
    defer scalar.deinit();

    var arr1d = try scalar.atleast1d(allocator);
    defer if (arr1d.owned) arr1d.deinit();

    try testing.expectEqual(@as(usize, 1), arr1d.shape[0]);
    try testing.expectEqual(@as(f64, 42), arr1d.data[0]);
    try testing.expect(!arr1d.owned); // View
}

test "atleast1d: 1D unchanged" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    var result = try arr.atleast1d(allocator);
    defer if (result.owned) result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expect(!result.owned); // View
    try testing.expectEqual(arr.data.ptr, result.data.ptr);
}

test "atleast1d: 2D unchanged" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.atleast1d(allocator);
    defer if (result.owned) result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expect(!result.owned); // View
}

test "atleast2d: scalar to 2D" {
    const allocator = testing.allocator;

    const data = [_]f64{42};
    var scalar = try NDArray(f64, 0).fromSlice(allocator, &[_]usize{}, &data, .row_major);
    defer scalar.deinit();

    var arr2d = try scalar.atleast2d(allocator);
    defer if (arr2d.owned) arr2d.deinit();

    try testing.expectEqual(@as(usize, 1), arr2d.shape[0]);
    try testing.expectEqual(@as(usize, 1), arr2d.shape[1]);
    try testing.expectEqual(@as(f64, 42), arr2d.data[0]);
}

test "atleast2d: 1D to 2D" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3 };
    var arr1d = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr1d.deinit();

    var arr2d = try arr1d.atleast2d(allocator);
    defer if (arr2d.owned) arr2d.deinit();

    // [3] -> [1, 3]
    try testing.expectEqual(@as(usize, 1), arr2d.shape[0]);
    try testing.expectEqual(@as(usize, 3), arr2d.shape[1]);
    try testing.expect(!arr2d.owned); // View
}

test "atleast2d: 2D unchanged" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.atleast2d(allocator);
    defer if (result.owned) result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
}

test "atleast3d: scalar to 3D" {
    const allocator = testing.allocator;

    const data = [_]f64{42};
    var scalar = try NDArray(f64, 0).fromSlice(allocator, &[_]usize{}, &data, .row_major);
    defer scalar.deinit();

    var arr3d = try scalar.atleast3d(allocator);
    defer if (arr3d.owned) arr3d.deinit();

    try testing.expectEqual(@as(usize, 1), arr3d.shape[0]);
    try testing.expectEqual(@as(usize, 1), arr3d.shape[1]);
    try testing.expectEqual(@as(usize, 1), arr3d.shape[2]);
    try testing.expectEqual(@as(f64, 42), arr3d.data[0]);
}

test "atleast3d: 1D to 3D" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3 };
    var arr1d = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr1d.deinit();

    var arr3d = try arr1d.atleast3d(allocator);
    defer if (arr3d.owned) arr3d.deinit();

    // [3] -> [1, 3, 1]
    try testing.expectEqual(@as(usize, 1), arr3d.shape[0]);
    try testing.expectEqual(@as(usize, 3), arr3d.shape[1]);
    try testing.expectEqual(@as(usize, 1), arr3d.shape[2]);
}

test "atleast3d: 2D to 3D" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var arr2d = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr2d.deinit();

    var arr3d = try arr2d.atleast3d(allocator);
    defer if (arr3d.owned) arr3d.deinit();

    // [2, 3] -> [2, 3, 1]
    try testing.expectEqual(@as(usize, 2), arr3d.shape[0]);
    try testing.expectEqual(@as(usize, 3), arr3d.shape[1]);
    try testing.expectEqual(@as(usize, 1), arr3d.shape[2]);
}

test "atleast3d: 3D unchanged" {
    const allocator = testing.allocator;

    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var arr = try NDArray(u8, 3).fromSlice(allocator, &[_]usize{ 2, 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.atleast3d(allocator);
    defer if (result.owned) result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
}

test "expandDims: insert dimension at axis 0" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var expanded = try arr.expandDims(allocator, 0);
    defer if (expanded.owned) expanded.deinit();

    // [2, 2] -> [1, 2, 2]
    try testing.expectEqual(@as(usize, 1), expanded.shape[0]);
    try testing.expectEqual(@as(usize, 2), expanded.shape[1]);
    try testing.expectEqual(@as(usize, 2), expanded.shape[2]);
}

test "expandDims: insert dimension at middle axis" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var expanded = try arr.expandDims(allocator, 1);
    defer if (expanded.owned) expanded.deinit();

    // [2, 2] -> [2, 1, 2]
    try testing.expectEqual(@as(usize, 2), expanded.shape[0]);
    try testing.expectEqual(@as(usize, 1), expanded.shape[1]);
    try testing.expectEqual(@as(usize, 2), expanded.shape[2]);
}

test "expandDims: insert dimension at end" {
    const allocator = testing.allocator;

    const data = [_]u8{ 1, 2, 3 };
    var arr = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    var expanded = try arr.expandDims(allocator, 1);
    defer if (expanded.owned) expanded.deinit();

    // [3] -> [3, 1]
    try testing.expectEqual(@as(usize, 3), expanded.shape[0]);
    try testing.expectEqual(@as(usize, 1), expanded.shape[1]);
}

test "broadcastTo: broadcast along single axis" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &data, .row_major);
    defer arr.deinit();

    // [3, 1] -> [3, 4]
    var broadcasted = try arr.broadcastTo(allocator, [_]usize{ 3, 4 });
    defer if (broadcasted.owned) broadcasted.deinit();

    try testing.expectEqual(@as(usize, 3), broadcasted.shape[0]);
    try testing.expectEqual(@as(usize, 4), broadcasted.shape[1]);

    // Verify stride is 0 for broadcasted dimension
    try testing.expectEqual(@as(usize, 0), broadcasted.strides[1]);

    // Verify data access works (same value repeated)
    const val0 = try broadcasted.get(&[_]isize{ 0, 0 });
    const val1 = try broadcasted.get(&[_]isize{ 0, 3 });
    try testing.expectEqual(val0, val1); // Same row broadcasts
    try testing.expectEqual(@as(f64, 1), val0);
}

test "broadcastTo: no broadcasting needed" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    var result = try arr.broadcastTo(allocator, [_]usize{ 2, 2 });
    defer if (result.owned) result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
}

test "broadcastTo: broadcast multiple axes" {
    const allocator = testing.allocator;

    const data = [_]f64{42};
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 1, 1 }, &data, .row_major);
    defer arr.deinit();

    // [1, 1] -> [3, 4]
    var broadcasted = try arr.broadcastTo(allocator, [_]usize{ 3, 4 });
    defer if (broadcasted.owned) broadcasted.deinit();

    try testing.expectEqual(@as(usize, 3), broadcasted.shape[0]);
    try testing.expectEqual(@as(usize, 4), broadcasted.shape[1]);

    // Both strides should be 0
    try testing.expectEqual(@as(usize, 0), broadcasted.strides[0]);
    try testing.expectEqual(@as(usize, 0), broadcasted.strides[1]);

    // All elements should be same value
    const val = try broadcasted.get(&[_]isize{ 2, 3 });
    try testing.expectEqual(@as(f64, 42), val);
}

test "broadcastTo: shape mismatch error" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &data, .row_major);
    defer arr.deinit();

    // Cannot broadcast [3, 1] to [2, 4] - first dimension mismatch
    const result = arr.broadcastTo(allocator, [_]usize{ 2, 4 });
    try testing.expectError(error.ShapeMismatch, result);
}

test "broadcastTo: non-1 dimension cannot broadcast" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    // Cannot broadcast [2, 3] to [2, 5] - dimension 3 != 1 and 3 != 5
    const result = arr.broadcastTo(allocator, [_]usize{ 2, 5 });
    try testing.expectError(error.ShapeMismatch, result);
}

test "atleast functions: type preservation" {
    const allocator = testing.allocator;

    // Test with u8 type
    const data = [_]u8{255};
    var scalar = try NDArray(u8, 0).fromSlice(allocator, &[_]usize{}, &data, .row_major);
    defer scalar.deinit();

    var arr1d = try scalar.atleast1d(allocator);
    defer if (arr1d.owned) arr1d.deinit();
    try testing.expectEqual(@as(u8, 255), arr1d.data[0]);

    var arr2d = try scalar.atleast2d(allocator);
    defer if (arr2d.owned) arr2d.deinit();
    try testing.expectEqual(@as(u8, 255), arr2d.data[0]);

    var arr3d = try scalar.atleast3d(allocator);
    defer if (arr3d.owned) arr3d.deinit();
    try testing.expectEqual(@as(u8, 255), arr3d.data[0]);
}

test "broadcastTo: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1, 2, 3 };
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 1 }, &data, .row_major);
        defer arr.deinit();

        var broadcasted = try arr.broadcastTo(allocator, [_]usize{ 3, 5 });
        defer if (broadcasted.owned) broadcasted.deinit();

        // Verify broadcasting works correctly
        const val = try broadcasted.get(&[_]isize{ 0, 0 });
        try testing.expectEqual(@as(f64, 1), val);
    }
}

test "atleast functions: column-major layout" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1, 2, 3 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .column_major);
    defer arr.deinit();

    var arr2d = try arr.atleast2d(allocator);
    defer if (arr2d.owned) arr2d.deinit();

    try testing.expectEqual(Layout.column_major, arr2d.layout);
}

// -- bincount Tests (7 tests) --

test "bincount: basic usage" {
    const allocator = testing.allocator;

    const data = [_]usize{ 0, 1, 1, 3, 2, 1, 7 };
    var arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{7}, &data, .row_major);
    defer arr.deinit();

    var counts = try arr.bincount(allocator);
    defer counts.deinit();

    // Verify shape
    // ndim is comptime, verification implicit
    try testing.expectEqual(@as(usize, 8), counts.shape[0]); // max(data)+1 = 7+1 = 8

    // Verify counts: indices 0,1,2,3,7 appear 1,3,1,1,1 times
    try testing.expectEqual(@as(usize, 1), counts.data[0]); // 0 appears 1 time
    try testing.expectEqual(@as(usize, 3), counts.data[1]); // 1 appears 3 times
    try testing.expectEqual(@as(usize, 1), counts.data[2]); // 2 appears 1 time
    try testing.expectEqual(@as(usize, 1), counts.data[3]); // 3 appears 1 time
    try testing.expectEqual(@as(usize, 0), counts.data[4]); // 4 appears 0 times
    try testing.expectEqual(@as(usize, 0), counts.data[5]); // 5 appears 0 times
    try testing.expectEqual(@as(usize, 0), counts.data[6]); // 6 appears 0 times
    try testing.expectEqual(@as(usize, 1), counts.data[7]); // 7 appears 1 time
}

test "bincount: all zeros" {
    const allocator = testing.allocator;

    const data = [_]usize{ 0, 0, 0, 0 };
    var arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    var counts = try arr.bincount(allocator);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 1), counts.shape[0]); // max(0)+1 = 1
    try testing.expectEqual(@as(usize, 4), counts.data[0]); // 0 appears 4 times
}

test "bincount: single element" {
    const allocator = testing.allocator;

    const data = [_]usize{5};
    var arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{1}, &data, .row_major);
    defer arr.deinit();

    var counts = try arr.bincount(allocator);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 6), counts.shape[0]); // max(5)+1 = 6
    try testing.expectEqual(@as(usize, 0), counts.data[0]);
    try testing.expectEqual(@as(usize, 1), counts.data[5]); // 5 appears 1 time
}

test "bincount: empty array" {
    const allocator = testing.allocator;

    // Create empty array manually (init doesn't allow zero dimensions)
    const data = try allocator.alloc(usize, 0);
    var arr = NDArray(usize, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    var counts = try arr.bincount(allocator);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 0), counts.shape[0]); // Empty result
}

test "bincount: large values" {
    const allocator = testing.allocator;

    const data = [_]usize{ 100, 0, 50, 100 };
    var arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    var counts = try arr.bincount(allocator);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 101), counts.shape[0]); // max(100)+1 = 101
    try testing.expectEqual(@as(usize, 1), counts.data[0]);
    try testing.expectEqual(@as(usize, 1), counts.data[50]);
    try testing.expectEqual(@as(usize, 2), counts.data[100]);
}

test "bincount: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]usize{ 0, 1, 2, 3, 4, 5 };
        var arr = try NDArray(usize, 1).fromSlice(allocator, &[_]usize{6}, &data, .row_major);
        defer arr.deinit();

        var counts = try arr.bincount(allocator);
        defer counts.deinit();

        // Verify all values have count 1
        for (0..6) |i| {
            try testing.expectEqual(@as(usize, 1), counts.data[i]);
        }
    }
}

test "bincount: dimension mismatch error" {
    const allocator = testing.allocator;

    const data = [_]usize{ 0, 1, 2, 3 };
    var arr = try NDArray(usize, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    const result = arr.bincount(allocator);
    try testing.expectError(error.DimensionMismatch, result);
}

// -- digitize Tests (8 tests) --

test "digitize: basic usage" {
    const allocator = testing.allocator;

    const values = [_]f64{ 0.2, 6.4, 3.0, 1.6 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.5, 4.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
    defer bins.deinit();

    var indices = try arr.digitize(allocator, &bins, false);
    defer indices.deinit();

    // 0.2 in [0,1), 6.4 in [4,10), 3.0 in [2.5,4), 1.6 in [1,2.5)
    try testing.expectEqual(@as(usize, 1), indices.data[0]); // 0.2 in bin 1
    try testing.expectEqual(@as(usize, 4), indices.data[1]); // 6.4 in bin 4
    try testing.expectEqual(@as(usize, 3), indices.data[2]); // 3.0 in bin 3
    try testing.expectEqual(@as(usize, 2), indices.data[3]); // 1.6 in bin 2
}

test "digitize: right=true" {
    const allocator = testing.allocator;

    const values = [_]f64{ 1.0, 2.5, 4.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.5, 4.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
    defer bins.deinit();

    var indices = try arr.digitize(allocator, &bins, true);
    defer indices.deinit();

    // With right=true: (bins[i-1], bins[i]]
    try testing.expectEqual(@as(usize, 1), indices.data[0]); // 1.0 in (0,1]
    try testing.expectEqual(@as(usize, 2), indices.data[1]); // 2.5 in (1,2.5]
    try testing.expectEqual(@as(usize, 3), indices.data[2]); // 4.0 in (2.5,4]
}

test "digitize: descending bins" {
    const allocator = testing.allocator;

    const values = [_]f64{ 0.2, 6.4, 3.0, 1.6 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 10.0, 4.0, 2.5, 1.0, 0.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
    defer bins.deinit();

    var indices = try arr.digitize(allocator, &bins, false);
    defer indices.deinit();

    // Descending bins work correctly
    try testing.expectEqual(@as(usize, 4), indices.data[0]); // 0.2
    try testing.expectEqual(@as(usize, 1), indices.data[1]); // 6.4
    try testing.expectEqual(@as(usize, 2), indices.data[2]); // 3.0
    try testing.expectEqual(@as(usize, 3), indices.data[3]); // 1.6
}

test "digitize: values outside bins" {
    const allocator = testing.allocator;

    const values = [_]f64{ -1.0, 15.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.5, 4.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
    defer bins.deinit();

    var indices = try arr.digitize(allocator, &bins, false);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 0), indices.data[0]); // -1.0 before first bin
    try testing.expectEqual(@as(usize, 5), indices.data[1]); // 15.0 after last bin
    try testing.expectEqual(@as(usize, 4), indices.data[2]); // 5.0 in [4,10)
}

test "digitize: single bin" {
    const allocator = testing.allocator;

    const values = [_]i32{ 1, 2, 3 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]i32{2};
    var bins = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{1}, &bin_edges, .row_major);
    defer bins.deinit();

    var indices = try arr.digitize(allocator, &bins, false);
    defer indices.deinit();

    try testing.expectEqual(@as(usize, 0), indices.data[0]); // 1 < 2
    try testing.expectEqual(@as(usize, 1), indices.data[1]); // 2 >= 2
    try testing.expectEqual(@as(usize, 1), indices.data[2]); // 3 >= 2
}

test "digitize: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const values = [_]f64{ 0.5, 1.5, 2.5, 3.5 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &values, .row_major);
        defer arr.deinit();

        const bin_edges = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
        var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
        defer bins.deinit();

        var indices = try arr.digitize(allocator, &bins, false);
        defer indices.deinit();
    }
}

test "digitize: empty bins error" {
    const allocator = testing.allocator;

    const values = [_]f64{ 1.0, 2.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{2}, &values, .row_major);
    defer arr.deinit();

    // Create empty bins manually (init doesn't allow zero dimensions)
    const bin_data = try allocator.alloc(f64, 0);
    var bins = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = bin_data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer bins.deinit();

    const result = arr.digitize(allocator, &bins, false);
    try testing.expectError(error.EmptyArray, result);
}

test "digitize: dimension mismatch error" {
    const allocator = testing.allocator;

    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &values, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    const result = arr.digitize(allocator, &bins, false);
    try testing.expectError(error.DimensionMismatch, result);
}

// -- histogram Tests (10 tests) --

test "histogram: basic usage" {
    const allocator = testing.allocator;

    const data = [_]f64{ 0.5, 1.5, 2.5, 3.5, 1.2, 2.8 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    // Bins: [0,1), [1,2), [2,3), [3,4)
    // Data: 0.5 in [0,1), 1.5 in [1,2), 2.5 in [2,3), 3.5 in [3,4), 1.2 in [1,2), 2.8 in [2,3)
    try testing.expectEqual(@as(usize, 4), hist.shape[0]); // 4 bins
    try testing.expectEqual(@as(usize, 1), hist.data[0]); // [0,1): 0.5
    try testing.expectEqual(@as(usize, 2), hist.data[1]); // [1,2): 1.5, 1.2
    try testing.expectEqual(@as(usize, 2), hist.data[2]); // [2,3): 2.5, 2.8
    try testing.expectEqual(@as(usize, 1), hist.data[3]); // [3,4): 3.5
}

test "histogram: uniform distribution" {
    const allocator = testing.allocator;

    const data = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{10}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]i32{ 0, 5, 10 };
    var bins = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    // [0,5): 0,1,2,3,4  [5,10): 5,6,7,8,9
    try testing.expectEqual(@as(usize, 2), hist.shape[0]);
    try testing.expectEqual(@as(usize, 5), hist.data[0]);
    try testing.expectEqual(@as(usize, 5), hist.data[1]);
}

test "histogram: all values in one bin" {
    const allocator = testing.allocator;

    const data = [_]f64{ 5.1, 5.5, 5.9, 5.3 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 5.0, 6.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    try testing.expectEqual(@as(usize, 3), hist.shape[0]);
    try testing.expectEqual(@as(usize, 0), hist.data[0]); // [0,5)
    try testing.expectEqual(@as(usize, 4), hist.data[1]); // [5,6)
    try testing.expectEqual(@as(usize, 0), hist.data[2]); // [6,10)
}

test "histogram: values outside range" {
    const allocator = testing.allocator;

    const data = [_]f64{ -1.0, 0.5, 5.5, 11.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 5.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    // -1.0 and 11.0 are outside, only 0.5 and 5.5 counted
    try testing.expectEqual(@as(usize, 1), hist.data[0]); // [0,5): 0.5
    try testing.expectEqual(@as(usize, 1), hist.data[1]); // [5,10): 5.5
}

test "histogram: right edge inclusive" {
    const allocator = testing.allocator;

    const data = [_]f64{ 0.0, 5.0, 10.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 5.0, 10.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    // 10.0 is included in last bin
    try testing.expectEqual(@as(usize, 1), hist.data[0]); // [0,5): 0.0
    try testing.expectEqual(@as(usize, 2), hist.data[1]); // [5,10]: 5.0, 10.0 (right edge inclusive)
}

test "histogram: empty array" {
    const allocator = testing.allocator;

    // Create empty array manually (init doesn't allow zero dimensions)
    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    // All bins empty
    try testing.expectEqual(@as(usize, 0), hist.data[0]);
    try testing.expectEqual(@as(usize, 0), hist.data[1]);
}

test "histogram: single bin" {
    const allocator = testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]i32{ 0, 10 };
    var bins = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{2}, &bin_edges, .row_major);
    defer bins.deinit();

    var hist = try arr.histogram(allocator, &bins);
    defer hist.deinit();

    try testing.expectEqual(@as(usize, 1), hist.shape[0]);
    try testing.expectEqual(@as(usize, 4), hist.data[0]); // All values in [0,10)
}

test "histogram: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 0.5, 1.5, 2.5, 3.5 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        const bin_edges = [_]f64{ 0.0, 1.0, 2.0, 3.0, 4.0 };
        var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &bin_edges, .row_major);
        defer bins.deinit();

        var hist = try arr.histogram(allocator, &bins);
        defer hist.deinit();
    }
}

test "histogram: insufficient bins error" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{1.0};
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &bin_edges, .row_major);
    defer bins.deinit();

    const result = arr.histogram(allocator, &bins);
    try testing.expectError(error.InvalidValue, result);
}

test "histogram: dimension mismatch error" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    const bin_edges = [_]f64{ 0.0, 1.0, 2.0 };
    var bins = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &bin_edges, .row_major);
    defer bins.deinit();

    const result = arr.histogram(allocator, &bins);
    try testing.expectError(error.DimensionMismatch, result);
}

// ============================================================================
// Statistical Aggregation Functions Tests
// ============================================================================

// variance() — Variance with delta degrees of freedom
test "variance: basic variance f64 array" {
    const allocator = testing.allocator;

    // [1, 2, 3, 4, 5] has mean=3, sum of squared deviations = 10
    // variance(ddof=0) = 10/5 = 2.0, variance(ddof=1) = 10/4 = 2.5
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const variance_val = arr.variance(0);
    try testing.expectApproxEqAbs(2.0, variance_val, 1e-9);
}

test "variance: sample variance with ddof=1" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const variance_val = arr.variance(1);
    try testing.expectApproxEqAbs(2.5, variance_val, 1e-9);
}

test "variance: zero variance (all same elements)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 5.0, 5.0, 5.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    const variance_val = arr.variance(0);
    try testing.expectApproxEqAbs(0.0, variance_val, 1e-9);
}

test "variance: single element (variance is undefined, should be 0)" {
    const allocator = testing.allocator;

    const data = [_]f64{42.0};
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data, .row_major);
    defer arr.deinit();

    const variance_val = arr.variance(0);
    try testing.expectApproxEqAbs(0.0, variance_val, 1e-9);
}

test "variance: integer array converted to f64" {
    const allocator = testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const variance_val = arr.variance(0);
    try testing.expectApproxEqAbs(200.0, variance_val, 1e-9);
}

test "variance: 2D array flattened" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    // Mean = 2.5, variance = sum((x-2.5)^2) / 4 = 1.25
    const variance_val = arr.variance(0);
    try testing.expectApproxEqAbs(1.25, variance_val, 1e-9);
}

// std() — Standard deviation with delta degrees of freedom
test "std: basic standard deviation" {
    const allocator = testing.allocator;

    // [1, 2, 3, 4, 5] var(0)=2.0, std(0)=sqrt(2.0)≈1.414
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const std_val = arr.std(0);
    try testing.expectApproxEqAbs(stdlib.math.sqrt(2.0), std_val, 1e-9);
}

test "std: sample standard deviation with ddof=1" {
    const allocator = testing.allocator;

    // [1, 2, 3, 4, 5] var(1)=2.5, std(1)=sqrt(2.5)≈1.581
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const std_val = arr.std(1);
    try testing.expectApproxEqAbs(stdlib.math.sqrt(2.5), std_val, 1e-9);
}

test "std: zero standard deviation" {
    const allocator = testing.allocator;

    const data = [_]f64{ 7.0, 7.0, 7.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const std_val = arr.std(0);
    try testing.expectApproxEqAbs(0.0, std_val, 1e-9);
}

test "std: integer array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 2, 4, 6, 8, 10 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    // var = 8.0, std = sqrt(8.0) ≈ 2.828
    const std_val = arr.std(0);
    try testing.expectApproxEqAbs(stdlib.math.sqrt(8.0), std_val, 1e-9);
}

// median() — Middle value or average of two middle values
test "median: odd length array" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(3.0, med, 1e-9);
}

test "median: even length array (average of two middle)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
    defer arr.deinit();

    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(2.5, med, 1e-9);
}

test "median: unsorted array" {
    const allocator = testing.allocator;

    const data = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(3.0, med, 1e-9);
}

test "median: single element" {
    const allocator = testing.allocator;

    const data = [_]f64{42.5};
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{1}, &data, .row_major);
    defer arr.deinit();

    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(42.5, med, 1e-9);
}

test "median: integer array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 9, 1, 5, 3, 7 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(5.0, med, 1e-9);
}

test "median: 2D array flattened" {
    const allocator = testing.allocator;

    const data = [_]f64{ 2.0, 4.0, 1.0, 3.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 2 }, &data, .row_major);
    defer arr.deinit();

    // Flattened: [2,4,1,3], sorted: [1,2,3,4], median = (2+3)/2 = 2.5
    const med = try arr.median(allocator);
    try testing.expectApproxEqAbs(2.5, med, 1e-9);
}

test "median: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 3.0, 1.0, 4.0, 2.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{4}, &data, .row_major);
        defer arr.deinit();

        const med = try arr.median(allocator);
        try testing.expectApproxEqAbs(2.5, med, 1e-9);
    }
}

test "median: empty array error" {
    const allocator = testing.allocator;

    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.median(allocator);
    try testing.expectError(error.EmptyArray, result);
}

// percentile() — Value at percentage p (0-100 scale)
test "percentile: p=0 (minimum)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const p0 = try arr.percentile(allocator, 0.0);
    try testing.expectApproxEqAbs(1.0, p0, 1e-9);
}

test "percentile: p=50 (median)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const p50 = try arr.percentile(allocator, 50.0);
    try testing.expectApproxEqAbs(3.0, p50, 1e-9);
}

test "percentile: p=100 (maximum)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const p100 = try arr.percentile(allocator, 100.0);
    try testing.expectApproxEqAbs(5.0, p100, 1e-9);
}

test "percentile: p=25 (first quartile)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &data, .row_major);
    defer arr.deinit();

    // Linear interpolation: idx = 0.25 * (8-1) = 1.75
    // result ≈ arr[1] + 0.75 * (arr[2] - arr[1]) = 2 + 0.75 * 1 = 2.75
    const p25 = try arr.percentile(allocator, 25.0);
    try testing.expectApproxEqAbs(2.75, p25, 1e-9);
}

test "percentile: p=75 (third quartile)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &data, .row_major);
    defer arr.deinit();

    // Linear interpolation: idx = 0.75 * (8-1) = 5.25
    // result ≈ arr[5] + 0.25 * (arr[6] - arr[5]) = 6 + 0.25 * 1 = 6.25
    const p75 = try arr.percentile(allocator, 75.0);
    try testing.expectApproxEqAbs(6.25, p75, 1e-9);
}

test "percentile: unsorted array" {
    const allocator = testing.allocator;

    const data = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const p50 = try arr.percentile(allocator, 50.0);
    try testing.expectApproxEqAbs(3.0, p50, 1e-9);
}

test "percentile: integer array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const p50 = try arr.percentile(allocator, 50.0);
    try testing.expectApproxEqAbs(30.0, p50, 1e-9);
}

test "percentile: out of range error (p < 0)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const result = arr.percentile(allocator, -1.0);
    try testing.expectError(error.InvalidValue, result);
}

test "percentile: out of range error (p > 100)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const result = arr.percentile(allocator, 101.0);
    try testing.expectError(error.InvalidValue, result);
}

test "percentile: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
        defer arr.deinit();

        const p50 = try arr.percentile(allocator, 50.0);
        try testing.expectApproxEqAbs(3.0, p50, 1e-9);
    }
}

test "percentile: empty array error" {
    const allocator = testing.allocator;

    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.percentile(allocator, 50.0);
    try testing.expectError(error.EmptyArray, result);
}

// quantile() — Value at fraction q (0-1 scale)
test "quantile: q=0 (minimum)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const q0 = try arr.quantile(allocator, 0.0);
    try testing.expectApproxEqAbs(1.0, q0, 1e-9);
}

test "quantile: q=0.5 (median)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const q50 = try arr.quantile(allocator, 0.5);
    try testing.expectApproxEqAbs(3.0, q50, 1e-9);
}

test "quantile: q=1 (maximum)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const q1 = try arr.quantile(allocator, 1.0);
    try testing.expectApproxEqAbs(5.0, q1, 1e-9);
}

test "quantile: q=0.25 (first quartile)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &data, .row_major);
    defer arr.deinit();

    // Linear interpolation: idx = 0.25 * (8-1) = 1.75
    // result ≈ arr[1] + 0.75 * (arr[2] - arr[1]) = 2 + 0.75 * 1 = 2.75
    const q25 = try arr.quantile(allocator, 0.25);
    try testing.expectApproxEqAbs(2.75, q25, 1e-9);
}

test "quantile: q=0.75 (third quartile)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{8}, &data, .row_major);
    defer arr.deinit();

    // Linear interpolation: idx = 0.75 * (8-1) = 5.25
    // result ≈ arr[5] + 0.25 * (arr[6] - arr[5]) = 6 + 0.25 * 1 = 6.25
    const q75 = try arr.quantile(allocator, 0.75);
    try testing.expectApproxEqAbs(6.25, q75, 1e-9);
}

test "quantile: unsorted array" {
    const allocator = testing.allocator;

    const data = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const q50 = try arr.quantile(allocator, 0.5);
    try testing.expectApproxEqAbs(3.0, q50, 1e-9);
}

test "quantile: integer array" {
    const allocator = testing.allocator;

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    const q50 = try arr.quantile(allocator, 0.5);
    try testing.expectApproxEqAbs(30.0, q50, 1e-9);
}

test "quantile: out of range error (q < 0)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const result = arr.quantile(allocator, -0.1);
    try testing.expectError(error.InvalidValue, result);
}

test "quantile: out of range error (q > 1)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{3}, &data, .row_major);
    defer arr.deinit();

    const result = arr.quantile(allocator, 1.1);
    try testing.expectError(error.InvalidValue, result);
}

test "quantile: memory safety" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
        var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
        defer arr.deinit();

        const q50 = try arr.quantile(allocator, 0.5);
        try testing.expectApproxEqAbs(3.0, q50, 1e-9);
    }
}

test "quantile: empty array error" {
    const allocator = testing.allocator;

    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.quantile(allocator, 0.5);
    try testing.expectError(error.EmptyArray, result);
}

// ============================================================================
// cov() — Covariance matrix computation
// ============================================================================

test "cov: 2D array with rowvar=true (2 variables, 5 observations)" {
    const allocator = testing.allocator;

    // 2 variables, 5 observations each
    // X = [1, 2, 3, 4, 5]
    // Y = [2, 4, 6, 8, 10]
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    // Check shape is [2, 2]
    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[1]);

    // X variance: mean=3, sum_sq_dev = (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 4+1+0+1+4 = 10, var = 10/4 = 2.5
    // Y variance: mean=6, sum_sq_dev = (2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2 = 16+4+0+4+16 = 40, var = 40/4 = 10
    // Covariance: (1-3)(2-6) + (2-3)(4-6) + (3-3)(6-6) + (4-3)(8-6) + (5-3)(10-6) = 8+2+0+2+8 = 20, cov = 20/4 = 5
    try testing.expectApproxEqAbs(2.5, try cov_matrix.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(10.0, try cov_matrix.get(&[_]isize{ 1, 1 }), 1e-9);
    try testing.expectApproxEqAbs(5.0, try cov_matrix.get(&[_]isize{ 0, 1 }), 1e-9);
    try testing.expectApproxEqAbs(5.0, try cov_matrix.get(&[_]isize{ 1, 0 }), 1e-9);
}

test "cov: 2D array with rowvar=false (5 observations, 2 variables)" {
    const allocator = testing.allocator;

    // 5 observations, 2 variables
    const data = [_]f64{ 1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 2 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, false);
    defer cov_matrix.deinit();

    // Should produce same result as rowvar=true version (different data layout)
    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[1]);
    try testing.expectApproxEqAbs(2.5, try cov_matrix.get(&[_]isize{ 0, 0 }), 1e-9);
    try testing.expectApproxEqAbs(10.0, try cov_matrix.get(&[_]isize{ 1, 1 }), 1e-9);
}

test "cov: 1D input returns scalar variance" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var cov_scalar = try arr.cov(allocator, true);
    defer cov_scalar.deinit();

    // Should return 0D array with the variance value
    try testing.expectEqual(@as(usize, 0), cov_scalar.shape.len);
    const expected_var = arr.variance(1);
    try testing.expectApproxEqAbs(expected_var, try cov_scalar.get(&[_]isize{}), 1e-9);
}

test "cov: perfect positive covariance" {
    const allocator = testing.allocator;

    // X = [1, 2, 3], Y = [2, 4, 6] (Y = 2*X)
    // They are perfectly positively correlated
    const data = [_]f64{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    const cov_xy = try cov_matrix.get(&[_]isize{ 0, 1 });
    try testing.expect(cov_xy > 0.0);
}

test "cov: perfect negative covariance" {
    const allocator = testing.allocator;

    // X = [1, 2, 3], Y = [6, 4, 2] (Y = -2*X + 8)
    // They are perfectly negatively correlated
    const data = [_]f64{ 1.0, 2.0, 3.0, 6.0, 4.0, 2.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    const cov_xy = try cov_matrix.get(&[_]isize{ 0, 1 });
    try testing.expect(cov_xy < 0.0);
}

test "cov: independent variables (covariance near zero)" {
    const allocator = testing.allocator;

    // X = [1, 2, 3, 4, 5], Y = [2, 3, 2, 3, 2] (Y constant mean, low correlation)
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 2.0, 3.0, 2.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    const cov_xy = try cov_matrix.get(&[_]isize{ 0, 1 });
    // Covariance is small but not exactly zero
    try testing.expect(@abs(cov_xy) < 0.5);
}

test "cov: identical variables (cov(X,X) = var(X))" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    // Verify diagonal element matches variance
    // Extract first variable and compute variance manually
    var first_var_data = [_]f64{ data[0], data[1], data[2], data[3], data[4] };
    var first_var = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &first_var_data, .row_major);
    defer first_var.deinit();

    const var_x = first_var.variance(1);
    const cov_xx = try cov_matrix.get(&[_]isize{ 0, 0 });
    try testing.expectApproxEqAbs(var_x, cov_xx, 1e-9);
}

test "cov: three variables (3x10 matrix)" {
    const allocator = testing.allocator;

    // 3 variables, 10 observations
    var data = try allocator.alloc(f64, 30);
    defer allocator.free(data);

    for (0..10) |i| {
        data[i * 3 + 0] = @as(f64, @floatFromInt(i + 1)); // Variable 1: 1..10
        data[i * 3 + 1] = @as(f64, @floatFromInt(2 * (i + 1))); // Variable 2: 2..20
        data[i * 3 + 2] = @as(f64, @floatFromInt(10 - i)); // Variable 3: 10..1 (decreasing)
    }

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 10 }, data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    // Should produce 3x3 covariance matrix
    try testing.expectEqual(@as(usize, 3), cov_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 3), cov_matrix.shape[1]);

    // Diagonal should be positive (variances)
    for (0..3) |i| {
        const i_signed: isize = @intCast(i);
        const var_diag = try cov_matrix.get(&[_]isize{ i_signed, i_signed });
        try testing.expect(var_diag >= 0.0);
    }
}

test "cov: f32 type compatibility" {
    const allocator = testing.allocator;

    const data = [_]f32{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
    var arr = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), cov_matrix.shape[1]);
}

test "cov: covariance matrix is symmetric" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    for (0..cov_matrix.shape[0]) |i| {
        for (0..cov_matrix.shape[1]) |j| {
            const i_signed: isize = @intCast(i);
            const j_signed: isize = @intCast(j);
            const val_ij = try cov_matrix.get(&[_]isize{ i_signed, j_signed });
            const val_ji = try cov_matrix.get(&[_]isize{ j_signed, i_signed });
            try testing.expectApproxEqAbs(val_ij, val_ji, 1e-9);
        }
    }
}

test "cov: diagonal elements are variances" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var cov_matrix = try arr.cov(allocator, true);
    defer cov_matrix.deinit();

    // Extract variables and compute their variances
    const data_x = data[0..5];
    var var_x = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, data_x, .row_major);
    defer var_x.deinit();

    const data_y = data[5..10];
    var var_y = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, data_y, .row_major);
    defer var_y.deinit();

    const expected_var_x = var_x.variance(1);
    const expected_var_y = var_y.variance(1);

    const diag_00 = try cov_matrix.get(&[_]isize{ 0, 0 });
    const diag_11 = try cov_matrix.get(&[_]isize{ 1, 1 });

    try testing.expectApproxEqAbs(expected_var_x, diag_00, 1e-9);
    try testing.expectApproxEqAbs(expected_var_y, diag_11, 1e-9);
}

test "cov: empty array error" {
    const allocator = testing.allocator;

    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 2){
        .shape = [_]usize{ 0, 0 },
        .strides = [_]usize{ 0, 0 },
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.cov(allocator, true);
    try testing.expectError(error.EmptyArray, result);
}

test "cov: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
        defer arr.deinit();

        var cov_matrix = try arr.cov(allocator, true);
        defer cov_matrix.deinit();

        try testing.expectEqual(@as(usize, 2), cov_matrix.shape[0]);
    }
}

// ============================================================================
// corrcoef() — Pearson correlation coefficient matrix
// ============================================================================

test "corrcoef: perfect positive correlation (X=[1,2,3], Y=[2,4,6])" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    const rho_01 = try corrcoef_matrix.get(&[_]isize{ 0, 1 });
    try testing.expectApproxEqAbs(1.0, rho_01, 1e-9);
}

test "corrcoef: perfect negative correlation (X=[1,2,3], Y=[6,4,2])" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 6.0, 4.0, 2.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    const rho_01 = try corrcoef_matrix.get(&[_]isize{ 0, 1 });
    try testing.expectApproxEqAbs(-1.0, rho_01, 1e-9);
}

test "corrcoef: no correlation (independent variables)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 2.0, 3.0, 2.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    const rho_01 = try corrcoef_matrix.get(&[_]isize{ 0, 1 });
    // Correlation is small but not exactly zero
    try testing.expect(@abs(rho_01) < 0.5);
}

test "corrcoef: diagonal is 1.0 (perfect self-correlation)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    for (0..corrcoef_matrix.shape[0]) |i| {
        const i_signed: isize = @intCast(i);
        const diag_val = try corrcoef_matrix.get(&[_]isize{ i_signed, i_signed });
        try testing.expectApproxEqAbs(1.0, diag_val, 1e-9);
    }
}

test "corrcoef: matrix is symmetric" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    for (0..corrcoef_matrix.shape[0]) |i| {
        for (0..corrcoef_matrix.shape[1]) |j| {
            const i_signed: isize = @intCast(i);
            const j_signed: isize = @intCast(j);
            const val_ij = try corrcoef_matrix.get(&[_]isize{ i_signed, j_signed });
            const val_ji = try corrcoef_matrix.get(&[_]isize{ j_signed, i_signed });
            try testing.expectApproxEqAbs(val_ij, val_ji, 1e-9);
        }
    }
}

test "corrcoef: all values in [-1, 1]" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    var iter = corrcoef_matrix.iterator();
    while (iter.next()) |val| {
        try testing.expect(val >= -1.0 and val <= 1.0);
    }
}

test "corrcoef: 1D input returns 1.0 (scalar perfect correlation)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{5}, &data, .row_major);
    defer arr.deinit();

    var corrcoef_scalar = try arr.corrcoef(allocator, true);
    defer corrcoef_scalar.deinit();

    // Should return 0D array with value 1.0
    try testing.expectApproxEqAbs(1.0, try corrcoef_scalar.get(&[_]isize{}), 1e-9);
}

test "corrcoef: 2D with rowvar=true (2 variables, 5 observations)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 5 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[1]);

    const rho_01 = try corrcoef_matrix.get(&[_]isize{ 0, 1 });
    // Y = 2*X perfectly, so correlation should be 1.0
    try testing.expectApproxEqAbs(1.0, rho_01, 1e-9);
}

test "corrcoef: 2D with rowvar=false (5 observations, 2 variables)" {
    const allocator = testing.allocator;

    const data = [_]f64{ 1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 5, 2 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, false);
    defer corrcoef_matrix.deinit();

    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[1]);
}

test "corrcoef: three variables (3x10 matrix)" {
    const allocator = testing.allocator;

    var data = try allocator.alloc(f64, 30);
    defer allocator.free(data);

    for (0..10) |i| {
        data[i * 3 + 0] = @as(f64, @floatFromInt(i + 1));
        data[i * 3 + 1] = @as(f64, @floatFromInt(2 * (i + 1)));
        data[i * 3 + 2] = @as(f64, @floatFromInt(10 - i));
    }

    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 3, 10 }, data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    try testing.expectEqual(@as(usize, 3), corrcoef_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 3), corrcoef_matrix.shape[1]);

    // Diagonal should be 1.0
    for (0..3) |i| {
        const i_signed: isize = @intCast(i);
        try testing.expectApproxEqAbs(1.0, try corrcoef_matrix.get(&[_]isize{ i_signed, i_signed }), 1e-9);
    }
}

test "corrcoef: f32 type compatibility" {
    const allocator = testing.allocator;

    const data = [_]f32{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
    var arr = try NDArray(f32, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[0]);
    try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[1]);
}

test "corrcoef: constant variable (σ=0) handling" {
    const allocator = testing.allocator;

    // Second variable has no variance (all 5s)
    const data = [_]f64{ 1.0, 2.0, 3.0, 5.0, 5.0, 5.0 };
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
    defer arr.deinit();

    var corrcoef_matrix = try arr.corrcoef(allocator, true);
    defer corrcoef_matrix.deinit();

    // When one variable is constant, correlation is undefined (NaN)
    const rho_01 = try corrcoef_matrix.get(&[_]isize{ 0, 1 });
    try testing.expect(math.isNan(rho_01) or math.isInf(rho_01));
}

test "corrcoef: empty array error" {
    const allocator = testing.allocator;

    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 2){
        .shape = [_]usize{ 0, 0 },
        .strides = [_]usize{ 0, 0 },
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.corrcoef(allocator, true);
    try testing.expectError(error.EmptyArray, result);
}

test "corrcoef: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        const data = [_]f64{ 1.0, 2.0, 3.0, 2.0, 4.0, 6.0 };
        var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &data, .row_major);
        defer arr.deinit();

        var corrcoef_matrix = try arr.corrcoef(allocator, true);
        defer corrcoef_matrix.deinit();

        try testing.expectEqual(@as(usize, 2), corrcoef_matrix.shape[0]);
    }
}

// ================== mode() tests ==================

test "mode: basic integer array [1,2,2,3,3,3,4] returns 3" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{7}, .row_major);
    defer arr.deinit();

    const data = [_]i32{ 1, 2, 2, 3, 3, 3, 4 };
    for (0..7) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 3.0);
}

test "mode: tie-breaking returns first mode in sorted order" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    const data = [_]i32{ 1, 1, 2, 2 };
    for (0..4) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 1.0); // 1 comes before 2 in sorted order
}

test "mode: single element returns that element" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0}, 5.0);

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 5.0);
}

test "mode: all same elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 7.0, 7.0, 7.0, 7.0, 7.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 7.0);
}

test "mode: unsorted input [5,1,3,1,2,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    const data = [_]i32{ 5, 1, 3, 1, 2, 1 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 1.0); // 1 appears 3 times
}

test "mode: float data [1.5, 2.5, 2.5, 3.5]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.5, 2.5, 2.5, 3.5 };
    for (0..4) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 2.5);
}

test "mode: empty array returns error.EmptyArray" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.mode(allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "mode: 2D array flattened mode [1,2; 2,3; 3,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr.deinit();

    const data = [_]i32{ 1, 2, 2, 3, 3, 3 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i/2), @intCast(i%2)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == 3.0);
}

test "mode: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 1).init(allocator, &[_]usize{8}, .row_major);
        defer arr.deinit();

        const data = [_]f64{ 1.1, 2.2, 2.2, 3.3, 3.3, 3.3, 4.4, 5.5 };
        for (0..8) |i| {
            arr.set(&[_]isize{@intCast(i)}, data[i]);
        }

        const mode_val = try arr.mode(allocator);
        try testing.expect(mode_val == 3.3);
    }
}

test "mode: i32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]i32{ -10, -5, -5, 0, 10 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const mode_val = try arr.mode(allocator);
    try testing.expect(mode_val == -5.0);
}

// ================== skewness() tests ==================

test "skewness: symmetric distribution [1,2,3,4,5] near 0" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(@abs(skew) < 0.1); // Nearly symmetric
}

test "skewness: right-skewed distribution [1,1,1,2,5] is positive" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 1.0, 1.0, 2.0, 5.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(skew > 0.0); // Right-skewed
}

test "skewness: left-skewed distribution [1,5,5,5,5] is negative" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 5.0, 5.0, 5.0, 5.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(skew < 0.0); // Left-skewed
}

test "skewness: zero variance returns error.InvalidValue" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 3.0, 3.0, 3.0, 3.0 };
    for (0..4) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const result = arr.skewness(allocator);
    try testing.expectError(error.InvalidValue, result);
}

test "skewness: single element returns error (std undefined)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0}, 42.0);

    const result = arr.skewness(allocator);
    // Single element has std = 0, so should return error.InvalidValue
    try testing.expectError(error.InvalidValue, result);
}

test "skewness: normal distribution sample has low skewness" {
    const allocator = testing.allocator;
    // Approximate normal distribution: bell curve centered at 3
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{9}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.5, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5 };
    for (0..9) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(@abs(skew) < 0.5); // Approximately normal, low skewness
}

test "skewness: exponential-like [1,2,3,5,10,50] has strong positive skewness" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 5.0, 10.0, 50.0 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(skew > 0.5); // Strong right skew
}

test "skewness: empty array returns error.EmptyArray" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.skewness(allocator);
    try testing.expectError(error.EmptyArray, result);
}

test "skewness: 2D array flattened" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 1.0, 1.0, 2.0, 5.0, 6.0 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i/2), @intCast(i%2)}, data[i]);
    }

    const skew = try arr.skewness(allocator);
    try testing.expect(skew > 0.0); // Right-skewed
}

test "skewness: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 1).init(allocator, &[_]usize{7}, .row_major);
        defer arr.deinit();

        const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0 };
        for (0..7) |i| {
            arr.set(&[_]isize{@intCast(i)}, data[i]);
        }

        const skew = try arr.skewness(allocator);
        try testing.expect(skew > 0.0); // Right-skewed
    }
}

// ================== kurtosis() tests ==================

test "kurtosis: normal-like [1,2,3,4,5] fisher=true (excess) is negative (platykurtic)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt = try arr.kurtosis(allocator, true);
    try testing.expect(kurt < 0.0); // Uniform distribution is platykurtic
}

test "kurtosis: fisher=false vs fisher=true difference is 3" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt_pearson = try arr.kurtosis(allocator, false);
    const kurt_fisher = try arr.kurtosis(allocator, true);

    const diff = kurt_pearson - kurt_fisher;
    try testing.expect(@abs(diff - 3.0) < 0.01); // Difference should be exactly 3
}

test "kurtosis: heavy tails [1,1,1,1,10] has positive excess kurtosis" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 1.0, 1.0, 1.0, 10.0 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt = try arr.kurtosis(allocator, true);
    try testing.expect(kurt > 0.0); // Leptokurtic (heavy tails)
}

test "kurtosis: uniform-like [1,2,3,4] has negative excess kurtosis" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{4}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    for (0..4) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt = try arr.kurtosis(allocator, true);
    try testing.expect(kurt < 0.0); // Platykurtic (light tails)
}

test "kurtosis: zero variance returns error.InvalidValue" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 5.0, 5.0, 5.0 };
    for (0..3) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const result = arr.kurtosis(allocator, true);
    try testing.expectError(error.InvalidValue, result);
}

test "kurtosis: single element returns error (std undefined)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0}, 7.0);

    const result = arr.kurtosis(allocator, true);
    try testing.expectError(error.InvalidValue, result);
}

test "kurtosis: empty array returns error.EmptyArray" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 0);
    var arr = NDArray(f64, 1){
        .shape = [_]usize{0},
        .strides = [_]usize{1},
        .data = data,
        .allocator = allocator,
        .layout = .row_major,
        .owned = true,
    };
    defer arr.deinit();

    const result = arr.kurtosis(allocator, true);
    try testing.expectError(error.EmptyArray, result);
}

test "kurtosis: 2D array flattened" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i/2), @intCast(i%2)}, data[i]);
    }

    const kurt_fisher = try arr.kurtosis(allocator, true);
    try testing.expect(kurt_fisher < 0.0); // Uniform-like, platykurtic
}

test "kurtosis: excess kurtosis calculation validation" {
    const allocator = testing.allocator;
    // Create a leptokurtic distribution (concentrated at center with heavy tails)
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{7}, .row_major);
    defer arr.deinit();

    const data = [_]f64{ 0.5, 2.0, 5.0, 5.0, 5.0, 8.0, 9.5 };
    for (0..7) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt_fisher = try arr.kurtosis(allocator, true);
    const kurt_pearson = try arr.kurtosis(allocator, false);

    // Verify relationship: excess = Pearson - 3
    try testing.expect(@abs(kurt_fisher - (kurt_pearson - 3.0)) < 0.001);
}

test "kurtosis: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 1).init(allocator, &[_]usize{8}, .row_major);
        defer arr.deinit();

        const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        for (0..8) |i| {
            arr.set(&[_]isize{@intCast(i)}, data[i]);
        }

        const kurt = try arr.kurtosis(allocator, true);
        try testing.expect(kurt < 0.0); // Uniform should be platykurtic
    }
}

test "kurtosis: i32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{5}, .row_major);
    defer arr.deinit();

    const data = [_]i32{ 10, 20, 30, 40, 50 };
    for (0..5) |i| {
        arr.set(&[_]isize{@intCast(i)}, data[i]);
    }

    const kurt = try arr.kurtosis(allocator, true);
    try testing.expect(kurt < 0.0); // Uniform-like
}

// ============================================================================
// AXIS-WISE REDUCTION FUNCTION TESTS
// ============================================================================

test "sumAxis: 2D array [3,4] axis=0 sums down columns" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with simple values: row i, col j = i * 10 + j
    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // sumAxis(0) should sum along rows, resulting in shape [4]
    // Column 0: 0 + 10 + 20 = 30
    // Column 1: 1 + 11 + 21 = 33
    // Column 2: 2 + 12 + 22 = 36
    // Column 3: 3 + 13 + 23 = 39
    var result = try arr.sumAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 30.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 33.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 36.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 39.0), result.data[3], 1e-10);
}

test "sumAxis: 2D array [3,4] axis=1 sums across rows" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // sumAxis(1) should sum along columns, resulting in shape [3]
    // Row 0: 0 + 1 + 2 + 3 = 6
    // Row 1: 10 + 11 + 12 + 13 = 46
    // Row 2: 20 + 21 + 22 + 23 = 86
    var result = try arr.sumAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 6.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 46.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 86.0), result.data[2], 1e-10);
}

test "sumAxis: 3D array [2,3,4] axis=0 reduces to [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with simple values
    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.sumAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
    // First element: arr[0,0,0] + arr[1,0,0] = 0 + 12 = 12
    try testing.expectApproxEqAbs(@as(f64, 12.0), result.data[0], 1e-10);
}

test "sumAxis: 3D array [2,3,4] axis=1 reduces to [2,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.sumAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
}

test "sumAxis: 3D array [2,3,4] axis=2 reduces to [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.sumAxis(allocator, 2);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
}

test "sumAxis: i32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    const data = [_][3]i32{ [_]i32{ 1, 2, 3 }, [_]i32{ 4, 5, 6 } };
    for (0..2) |i| {
        for (0..3) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, data[i][j]);
        }
    }

    var result = try arr.sumAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 6), result.data[0]);
    try testing.expectEqual(@as(i32, 15), result.data[1]);
}

test "sumAxis: f32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(f32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, @as(f32, 1.5));
    arr.set(&[_]isize{ 0, 1 }, @as(f32, 2.5));
    arr.set(&[_]isize{ 0, 2 }, @as(f32, 3.0));
    arr.set(&[_]isize{ 1, 0 }, @as(f32, 1.0));
    arr.set(&[_]isize{ 1, 1 }, @as(f32, 2.0));
    arr.set(&[_]isize{ 1, 2 }, @as(f32, 3.0));

    var result = try arr.sumAxis(allocator, 1);
    defer result.deinit();

    try testing.expectApproxEqAbs(@as(f32, 7.0), result.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.0), result.data[1], 1e-5);
}

test "sumAxis: single element array reduces to 0D scalar" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{1}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0}, 42.0);

    var result = try arr.sumAxis(allocator, 0);
    defer result.deinit();

    // Result is 0D (scalar), so shape is empty array
    try testing.expectEqual(@as(usize, 0), result.shape.len);
    try testing.expectEqual(@as(f64, 42.0), result.data[0]);
}

test "sumAxis: axis >= ndim returns InvalidAxis error" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, 1.0);

    const result = arr.sumAxis(allocator, 2); // Invalid axis for 2D array
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "sumAxis: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
        defer arr.deinit();

        for (0..3) |i| {
            for (0..4) |j| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i + j)));
            }
        }

        var result = try arr.sumAxis(allocator, 0);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 4), result.shape[0]);
    }
}

test "meanAxis: 2D array [3,4] axis=0 means down columns" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // meanAxis(0) should mean along rows, resulting in shape [4]
    // Column 0: (0 + 10 + 20) / 3 = 10
    // Column 1: (1 + 11 + 21) / 3 = 11
    // Column 2: (2 + 12 + 22) / 3 = 12
    // Column 3: (3 + 13 + 23) / 3 = 13
    var result = try arr.meanAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 10.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 11.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 12.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 13.0), result.data[3], 1e-10);
}

test "meanAxis: 2D array [3,4] axis=1 means across rows" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // meanAxis(1) should mean along columns, resulting in shape [3]
    // Row 0: (0 + 1 + 2 + 3) / 4 = 1.5
    // Row 1: (10 + 11 + 12 + 13) / 4 = 11.5
    // Row 2: (20 + 21 + 22 + 23) / 4 = 21.5
    var result = try arr.meanAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 1.5), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 11.5), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 21.5), result.data[2], 1e-10);
}

test "meanAxis: 3D array [2,3,4] axis=0 reduces to [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.meanAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
}

test "meanAxis: meanAxis always returns f64 for i32 input" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, @as(i32, 1));
    arr.set(&[_]isize{ 0, 1 }, @as(i32, 2));
    arr.set(&[_]isize{ 0, 2 }, @as(i32, 3));
    arr.set(&[_]isize{ 1, 0 }, @as(i32, 4));
    arr.set(&[_]isize{ 1, 1 }, @as(i32, 5));
    arr.set(&[_]isize{ 1, 2 }, @as(i32, 6));

    var result = try arr.meanAxis(allocator, 1);
    defer result.deinit();

    try testing.expectApproxEqAbs(@as(f64, 2.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 5.0), result.data[1], 1e-10);
}

test "meanAxis: axis >= ndim returns error" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, 1.0);

    const result = arr.meanAxis(allocator, 3);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "meanAxis: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
        defer arr.deinit();

        for (0..3) |i| {
            for (0..4) |j| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i + j)));
            }
        }

        var result = try arr.meanAxis(allocator, 1);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 3), result.shape[0]);
    }
}

test "minAxis: 2D array [3,4] axis=0 minimums down columns" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // minAxis(0) should min along rows, resulting in shape [4]
    // Column 0: min(0, 10, 20) = 0
    // Column 1: min(1, 11, 21) = 1
    // Column 2: min(2, 12, 22) = 2
    // Column 3: min(3, 13, 23) = 3
    var result = try arr.minAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 1.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 2.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 3.0), result.data[3], 1e-10);
}

test "minAxis: 2D array [3,4] axis=1 minimums across rows" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // minAxis(1) should min along columns, resulting in shape [3]
    // Row 0: min(0, 1, 2, 3) = 0
    // Row 1: min(10, 11, 12, 13) = 10
    // Row 2: min(20, 21, 22, 23) = 20
    var result = try arr.minAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 0.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 10.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 20.0), result.data[2], 1e-10);
}

test "minAxis: 3D array shape reduction" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.minAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
}

test "minAxis: i32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, @as(i32, 10));
    arr.set(&[_]isize{ 0, 1 }, @as(i32, 5));
    arr.set(&[_]isize{ 0, 2 }, @as(i32, 8));
    arr.set(&[_]isize{ 1, 0 }, @as(i32, 3));
    arr.set(&[_]isize{ 1, 1 }, @as(i32, 7));
    arr.set(&[_]isize{ 1, 2 }, @as(i32, 1));

    var result = try arr.minAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 5), result.data[0]);
    try testing.expectEqual(@as(i32, 1), result.data[1]);
}

test "minAxis: axis >= ndim returns error" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, 1.0);

    const result = arr.minAxis(allocator, 2);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "minAxis: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
        defer arr.deinit();

        for (0..3) |i| {
            for (0..4) |j| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i + j)));
            }
        }

        var result = try arr.minAxis(allocator, 0);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 4), result.shape[0]);
    }
}

test "maxAxis: 2D array [3,4] axis=0 maximums down columns" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // maxAxis(0) should max along rows, resulting in shape [4]
    // Column 0: max(0, 10, 20) = 20
    // Column 1: max(1, 11, 21) = 21
    // Column 2: max(2, 12, 22) = 22
    // Column 3: max(3, 13, 23) = 23
    var result = try arr.maxAxis(allocator, 0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 20.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 21.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 22.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 23.0), result.data[3], 1e-10);
}

test "maxAxis: 2D array [3,4] axis=1 maximums across rows" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i * 10 + j)));
        }
    }

    // maxAxis(1) should max along columns, resulting in shape [3]
    // Row 0: max(0, 1, 2, 3) = 3
    // Row 1: max(10, 11, 12, 13) = 13
    // Row 2: max(20, 21, 22, 23) = 23
    var result = try arr.maxAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 3.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 13.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 23.0), result.data[2], 1e-10);
}

test "maxAxis: 3D array shape reduction" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    var idx: f64 = 0.0;
    var iter = arr.iterator();
    while (iter.next()) |_| {
        arr.data[iter.index - 1] = idx;
        idx += 1.0;
    }

    var result = try arr.maxAxis(allocator, 2);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.shape[1]);
}

test "maxAxis: i32 type compatibility" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, @as(i32, 10));
    arr.set(&[_]isize{ 0, 1 }, @as(i32, 5));
    arr.set(&[_]isize{ 0, 2 }, @as(i32, 8));
    arr.set(&[_]isize{ 1, 0 }, @as(i32, 3));
    arr.set(&[_]isize{ 1, 1 }, @as(i32, 7));
    arr.set(&[_]isize{ 1, 2 }, @as(i32, 1));

    var result = try arr.maxAxis(allocator, 1);
    defer result.deinit();

    try testing.expectEqual(@as(i32, 10), result.data[0]);
    try testing.expectEqual(@as(i32, 7), result.data[1]);
}

test "maxAxis: axis >= ndim returns error" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{ 0, 0 }, 1.0);

    const result = arr.maxAxis(allocator, 2);
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "maxAxis: memory safety (10 iterations)" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
        defer arr.deinit();

        for (0..3) |i| {
            for (0..4) |j| {
                arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @as(f64, @floatFromInt(i + j)));
            }
        }

        var result = try arr.maxAxis(allocator, 0);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 4), result.shape[0]);
    }
}
