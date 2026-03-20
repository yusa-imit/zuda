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
const testing = std.testing;
const Allocator = std.mem.Allocator;

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
        pub fn init(allocator: Allocator, shape: []const usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
                if (total_elements > std.math.maxInt(usize) / dim) {
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
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) deallocation
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
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
                std.debug.assert(dim > 0);
            }

            // Check data length matches product of shape
            const expected_len = self.count();
            std.debug.assert(self.data.len == expected_len);

            // Verify strides are consistent with shape and layout
            const expected_strides = calculateStrides(self.shape, self.layout);
            for (0..ndim) |i| {
                std.debug.assert(self.strides[i] == expected_strides[i]);
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
        pub fn zeros(allocator: Allocator, shape: []const usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn ones(allocator: Allocator, shape: []const usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn full(allocator: Allocator, shape: []const usize, value: T, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn empty(allocator: Allocator, shape: []const usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn arange(allocator: Allocator, start: T, stop: T, step: T, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
            // Validate step is not zero
            if (step == 0) {
                return error.ZeroDimension;
            }

            // Calculate number of elements
            var num_elements: usize = 0;
            if (step > 0) {
                if (start < stop) {
                    const diff = @as(f64, @floatFromInt(stop - start)) / @as(f64, @floatFromInt(step));
                    num_elements = @as(usize, @intFromFloat(@ceil(diff)));
                }
            } else {
                if (start > stop) {
                    const diff = @as(f64, @floatFromInt(start - stop)) / @as(f64, @floatFromInt(-step));
                    num_elements = @as(usize, @intFromFloat(@ceil(diff)));
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
        pub fn linspace(allocator: Allocator, start: T, stop: T, num: usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn fromSlice(allocator: Allocator, shape: []const usize, data_slice: []const T, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
            var arr = try Self.init(allocator, shape, layout);
            errdefer arr.deinit();

            if (arr.data.len != data_slice.len) {
                return error.CapacityExceeded;
            }

            @memcpy(arr.data, data_slice);
            return arr;
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
        pub fn eye(allocator: Allocator, rows: usize, cols: usize, k: isize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
            var arr = try Self.init(allocator, &[_]usize{rows, cols}, layout);
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
        pub fn identity(allocator: Allocator, rows: usize, cols: usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
            return Self.eye(allocator, rows, cols, 0, layout);
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
            };
        }
    };
}

// ============================================================================
// TESTS — Following TDD "Red" phase (all must FAIL before implementation)
// ============================================================================

// -- Type Definition Tests (3 tests) --

test "ndarray: NDArray(f64, 2) type creation" {
    const ArrayType = NDArray(f64, 2);
    const allocator = testing.allocator;

    // Verify type has required fields
    _ = allocator;
    const arr: ArrayType = undefined;
    _ = &arr.shape;
    _ = &arr.strides;
    _ = &arr.data;
    _ = &arr.allocator;
    _ = &arr.layout;
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
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Should allocate 3*4 = 12 elements
    try testing.expectEqual(12, arr.data.len);
    try testing.expectEqual(12, arr.count());
    try testing.expectEqual(3, arr.shape[0]);
    try testing.expectEqual(4, arr.shape[1]);
}

test "ndarray: init row-major stride calculation [3,4] → [4,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Row-major [3,4]: strides should be [4, 1] (next row = 4 elements)
    try testing.expectEqual(4, arr.strides[0]);
    try testing.expectEqual(1, arr.strides[1]);
}

test "ndarray: init column-major stride calculation [3,4] → [1,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .column_major);
    defer arr.deinit();

    // Column-major [3,4]: strides should be [1, 3] (next column = 3 elements)
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(3, arr.strides[1]);
}

test "ndarray: init rejects zero-sized dimensions" {
    const allocator = testing.allocator;
    const result = NDArray(i32, 2).init(allocator, &[_]usize{3, 0}, .row_major);

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: init rejects oversized shape exceeding usize max" {
    const allocator = testing.allocator;
    // Attempt: [usize.max, 2] → product overflows
    const result = NDArray(f64, 2).init(allocator, &[_]usize{std.math.maxInt(usize), 2}, .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

// -- Memory Layout Tests (6 tests) --

test "ndarray: 3D row-major [2,3,4] strides [12,4,1]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr.deinit();

    // Row-major: stride[i] = prod(shape[i+1..])
    // [2,3,4] → [3*4, 4, 1] = [12, 4, 1]
    try testing.expectEqual(12, arr.strides[0]);
    try testing.expectEqual(4, arr.strides[1]);
    try testing.expectEqual(1, arr.strides[2]);
}

test "ndarray: 3D column-major [2,3,4] strides [1,2,6]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .column_major);
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
    var arr = try NDArray(u8, 4).init(allocator, &[_]usize{2, 3, 4, 5}, .row_major);
    defer arr.deinit();

    // [2,3,4,5] → [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1]
    try testing.expectEqual(60, arr.strides[0]);
    try testing.expectEqual(20, arr.strides[1]);
    try testing.expectEqual(5, arr.strides[2]);
    try testing.expectEqual(1, arr.strides[3]);
}

test "ndarray: 4D column-major [2,3,4,5] strides [1,2,6,24]" {
    const allocator = testing.allocator;
    var arr = try NDArray(u8, 4).init(allocator, &[_]usize{2, 3, 4, 5}, .column_major);
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
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{100, 100}, .row_major);
    arr.deinit();

    // std.testing.allocator automatically detects leaks
    // If deinit didn't free, test will fail with leak error
}

test "ndarray: multiple init/deinit cycles don't leak" {
    const allocator = testing.allocator;

    for (0..10) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{5, 6, 7}, .row_major);
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

    // Should not crash (double-free protection)
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
    const shape = [_]usize{2, 3, 4};
    var arr = try NDArray(i32, 3).init(allocator, shape[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(2, arr.shape[0]);
    try testing.expectEqual(3, arr.shape[1]);
    try testing.expectEqual(4, arr.shape[2]);
}

test "ndarray: layout stored correctly" {
    const allocator = testing.allocator;

    var arr_rm = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr_rm.deinit();
    try testing.expectEqual(Layout.row_major, arr_rm.layout);

    var arr_cm = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .column_major);
    defer arr_cm.deinit();
    try testing.expectEqual(Layout.column_major, arr_cm.layout);
}

// -- Count and isEmpty Tests (3 tests) --

test "ndarray: count() returns prod(shape)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
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

    var arr = try NDArray(i32, 4).init(allocator, &[_]usize{1, 1, 1, 1}, .row_major);
    defer arr.deinit();

    try testing.expect(!arr.isEmpty());
}

// -- Validation Tests (3 tests) --

test "ndarray: validate() checks dimension invariant" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Should not error on valid array
    try arr.validate();
}

test "ndarray: validate() checks stride consistency with row-major" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 5}, .row_major);
    defer arr.deinit();

    // Row-major [3,5] must have strides [5,1]
    // After init, validate should pass
    try arr.validate();
}

test "ndarray: validate() checks stride consistency with column-major" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 5}, .column_major);
    defer arr.deinit();

    // Column-major [3,5] must have strides [1,3]
    // After init, validate should pass
    try arr.validate();
}

// -- Allocator Storage Tests (2 tests) --

test "ndarray: allocator is stored and accessible" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{2, 2}, .row_major);
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
    const shape = [_]usize{3, 4};
    var arr = try NDArray(f64, 2).init(allocator, shape[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.data.len);
}

test "ndarray: data is contiguous slice of correct length" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
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
    var arr2 = try NDArray(f64, 2).init(allocator, &[_]usize{10, 10}, .row_major);
    defer arr2.deinit();
    try testing.expectEqual(100, arr2.count());

    // Test 3D
    var arr3 = try NDArray(f64, 3).init(allocator, &[_]usize{5, 5, 4}, .row_major);
    defer arr3.deinit();
    try testing.expectEqual(100, arr3.count());

    // Test 5D
    var arr5 = try NDArray(f64, 5).init(allocator, &[_]usize{2, 2, 5, 5, 1}, .row_major);
    defer arr5.deinit();
    try testing.expectEqual(100, arr5.count());
}

test "ndarray: stress test — layout consistency across dimensions" {
    const allocator = testing.allocator;

    // Verify row-major strides for various shapes
    for (0..5) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
        defer arr.deinit();
        try testing.expectEqual(12, arr.strides[0]);
        try testing.expectEqual(4, arr.strides[1]);
        try testing.expectEqual(1, arr.strides[2]);
    }

    // Verify column-major strides for various shapes
    for (0..5) |_| {
        var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .column_major);
        defer arr.deinit();
        try testing.expectEqual(1, arr.strides[0]);
        try testing.expectEqual(2, arr.strides[1]);
        try testing.expectEqual(6, arr.strides[2]);
    }
}

// -- Different Element Types Tests (3 tests) --

test "ndarray: f32 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(f32, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(6, arr.count());
}

test "ndarray: u64 element type" {
    const allocator = testing.allocator;
    var arr = try NDArray(u64, 2).init(allocator, &[_]usize{4, 5}, .row_major);
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
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() creates 3D array [2,3,4] with all zeros" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).zeros(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(24, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(0.0, val);
    }
}

test "ndarray: zeros() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .column_major);
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
    const result = NDArray(f64, 2).zeros(allocator, &[_]usize{3, 0}, .row_major);

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
    var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() creates 3D array [2,3,4] with all ones" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).ones(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(24, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).ones(allocator, &[_]usize{3, 4}, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    for (arr.data) |val| {
        try testing.expectEqual(1.0, val);
    }
}

test "ndarray: ones() works with u8 type" {
    const allocator = testing.allocator;
    var arr = try NDArray(u8, 2).ones(allocator, &[_]usize{2, 3}, .row_major);
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
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{3, 4}, 42.0, .row_major);
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
    var arr = try NDArray(i32, 2).full(allocator, &[_]usize{2, 3}, -100, .row_major);
    defer arr.deinit();

    try testing.expectEqual(6, arr.count());
    for (arr.data) |val| {
        try testing.expectEqual(@as(i32, -100), val);
    }
}

test "ndarray: full() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{3, 4}, 7.5, .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    for (arr.data) |val| {
        try testing.expectEqual(7.5, val);
    }
}

test "ndarray: full() works with 0.0 value" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{2, 2}, 0.0, .row_major);
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
    var arr = try NDArray(f64, 2).empty(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    try testing.expectEqual(3, arr.shape[0]);
    try testing.expectEqual(4, arr.shape[1]);
}

test "ndarray: empty() respects column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).empty(allocator, &[_]usize{3, 4}, .column_major);
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
    const expected = [_]f64{0.0, 2.0, 4.0, 6.0, 8.0};
    for (0..5) |i| {
        try testing.expectEqual(expected[i], arr.data[i]);
    }
}

test "ndarray: arange() creates [5, 15) with step 3" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 5, 15, 3, .row_major);
    defer arr.deinit();

    try testing.expectEqual(4, arr.count());
    const expected = [_]i32{5, 8, 11, 14};
    for (0..4) |i| {
        try testing.expectEqual(expected[i], arr.data[i]);
    }
}

test "ndarray: arange() handles descending range [10, 0) with step -1" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 10, 0, -1, .row_major);
    defer arr.deinit();

    try testing.expectEqual(10, arr.count());
    for (0..10) |i| {
        try testing.expectEqual(@as(i32, @intCast(10 - i - 1)), arr.data[i]);
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

// -- fromSlice() Creation Function Tests (5 tests) --

test "ndarray: fromSlice() creates 2D array [3,4] from slice" {
    const allocator = testing.allocator;
    const data = [_]f64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, data[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(12, arr.count());
    for (0..12) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), arr.data[i]);
    }
}

test "ndarray: fromSlice() creates 1D array [5]" {
    const allocator = testing.allocator;
    const data = [_]i32{10, 20, 30, 40, 50};
    var arr = try NDArray(i32, 1).fromSlice(allocator, &[_]usize{5}, data[0..], .row_major);
    defer arr.deinit();

    try testing.expectEqual(5, arr.count());
    for (0..5) |i| {
        try testing.expectEqual(data[i], arr.data[i]);
    }
}

test "ndarray: fromSlice() respects column-major layout [2,3]" {
    const allocator = testing.allocator;
    const data = [_]f64{1, 2, 3, 4, 5, 6};
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, data[0..], .column_major);
    defer arr.deinit();

    try testing.expectEqual(Layout.column_major, arr.layout);
    try testing.expectEqual(1, arr.strides[0]);
    try testing.expectEqual(2, arr.strides[1]);
}

test "ndarray: fromSlice() rejects shape mismatch" {
    const allocator = testing.allocator;
    const data = [_]f64{1, 2, 3};
    // Try to create 2D array [3,4] (12 elements) from 3-element slice
    const result = NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, data[0..], .row_major);

    try testing.expectError(error.CapacityExceeded, result);
}

test "ndarray: fromSlice() creates 3D array [2,2,2]" {
    const allocator = testing.allocator;
    const data = [_]i32{1, 2, 3, 4, 5, 6, 7, 8};
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{2, 2, 2}, data[0..], .row_major);
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

// -- Indexing Tests (get/set) (7 tests) --

test "ndarray: get() retrieves single element from 2D array [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{1, 2, 3, 4, 5, 6}, .row_major);
    defer arr.deinit();

    // Row-major [2,3]: [1,2,3 | 4,5,6]
    // Get [0,0] = 1.0, [0,2] = 3.0, [1,1] = 5.0
    try testing.expectEqual(1.0, arr.get(&[_]isize{0, 0}));
    try testing.expectEqual(3.0, arr.get(&[_]isize{0, 2}));
    try testing.expectEqual(5.0, arr.get(&[_]isize{1, 1}));
}

test "ndarray: get() supports negative indexing (-1 = last element)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{1, 2, 3, 4, 5, 6}, .row_major);
    defer arr.deinit();

    // [-1,-1] = [1,2] (last row, last col) = 6.0
    try testing.expectEqual(6.0, arr.get(&[_]isize{-1, -1}));
    // [-1,0] = [1,0] (last row, first col) = 4.0
    try testing.expectEqual(4.0, arr.get(&[_]isize{-1, 0}));
    // [0,-1] = [0,2] (first row, last col) = 3.0
    try testing.expectEqual(3.0, arr.get(&[_]isize{0, -1}));
    // [-2,-2] = [0,1] (second-to-last row, second-to-last col) = 2.0
    try testing.expectEqual(2.0, arr.get(&[_]isize{-2, -2}));
}

test "ndarray: get() retrieves from 3D array [2,3,4]" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{2, 3, 4}, data[0..], .row_major);
    defer arr.deinit();

    // Row-major [2,3,4]: element [i,j,k] at index i*12 + j*4 + k
    try testing.expectEqual(@as(i32, 0), arr.get(&[_]isize{0, 0, 0}));
    try testing.expectEqual(@as(i32, 3), arr.get(&[_]isize{0, 0, 3}));
    try testing.expectEqual(@as(i32, 12), arr.get(&[_]isize{1, 0, 0}));
    try testing.expectEqual(@as(i32, 23), arr.get(&[_]isize{1, 2, 3}));
}

test "ndarray: set() modifies single element in 2D array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0, 1}, 42.0);
    arr.set(&[_]isize{1, 2}, 99.0);

    try testing.expectEqual(42.0, arr.data[1]);
    try testing.expectEqual(99.0, arr.data[5]);
    try testing.expectEqual(0.0, arr.data[0]);
}

test "ndarray: set() with negative indices modifies last row/col" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{-1, -1}, 100);
    arr.set(&[_]isize{-2, 0}, 50);

    // [-1,-1] = [2,3] at index 2*4 + 3 = 11
    try testing.expectEqual(@as(i32, 100), arr.data[11]);
    // [-2,0] = [1,0] at index 1*4 + 0 = 4
    try testing.expectEqual(@as(i32, 50), arr.data[4]);
}

test "ndarray: set() persists changes across multiple operations" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).full(allocator, &[_]usize{2, 2}, 1.0, .row_major);
    defer arr.deinit();

    arr.set(&[_]isize{0, 0}, 10.0);
    arr.set(&[_]isize{0, 1}, 20.0);
    arr.set(&[_]isize{1, 0}, 30.0);
    arr.set(&[_]isize{1, 1}, 40.0);

    try testing.expectEqual(10.0, arr.get(&[_]isize{0, 0}));
    try testing.expectEqual(20.0, arr.get(&[_]isize{0, 1}));
    try testing.expectEqual(30.0, arr.get(&[_]isize{1, 0}));
    try testing.expectEqual(40.0, arr.get(&[_]isize{1, 1}));
}

test "ndarray: get() rejects out-of-bounds positive indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Shape [2,3] allows indices [0-1, 0-2]
    // [2,0] is out of bounds (row index too high)
    try testing.expectError(error.IndexOutOfBounds, arr.get(&[_]isize{2, 0}));
}

test "ndarray: get() rejects out-of-bounds negative indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // [-3, 0] is out of bounds for shape [2,3] (would be row -3)
    try testing.expectError(error.IndexOutOfBounds, arr.get(&[_]isize{-3, 0}));
}

// -- Flat Indexing Tests (at) (6 tests) --

test "ndarray: at() returns element at flat index in row-major [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{1, 2, 3, 4, 5, 6}, .row_major);
    defer arr.deinit();

    // Flat indexing: 0→1, 1→2, 2→3, 3→4, 4→5, 5→6
    try testing.expectEqual(1.0, arr.at(0));
    try testing.expectEqual(3.0, arr.at(2));
    try testing.expectEqual(4.0, arr.at(3));
    try testing.expectEqual(6.0, arr.at(5));
}

test "ndarray: at() supports negative flat indices (-1 = last element)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, &[_]f64{1, 2, 3, 4, 5, 6}, .row_major);
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
    const data = [_]f64{1, 2, 3, 4, 5, 6};

    var arr_rm = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, data[0..], .row_major);
    defer arr_rm.deinit();

    var arr_cm = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{2, 3}, data[0..], .column_major);
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
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{2, 3, 4}, data[0..], .row_major);
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
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    // Slice row 1: [5, 6, 7, 8]
    const sliced = arr.slice(&[_][2]?isize{
        .{ 1, 2 },  // rows 1:2 (single row)
        .{ null, null }, // all columns
    });

    try testing.expectEqual(@as(usize, 1), sliced.shape[0]);
    try testing.expectEqual(@as(usize, 4), sliced.shape[1]);
    try testing.expectEqual(5.0, sliced.at(0));
    try testing.expectEqual(8.0, sliced.at(3));
}

test "ndarray: slice() extracts column from 2D array [3,4]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
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
    try testing.expectEqual(3.0, sliced.at(0));
    try testing.expectEqual(7.0, sliced.at(1));
    try testing.expectEqual(11.0, sliced.at(2));
}

test "ndarray: slice() extracts rectangular subregion [3,4] → [2,2]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
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
    try testing.expectEqual(6.0, sliced.get(&[_]isize{0, 0}));
    try testing.expectEqual(7.0, sliced.get(&[_]isize{0, 1}));
    try testing.expectEqual(10.0, sliced.get(&[_]isize{1, 0}));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{1, 1}));
}

test "ndarray: slice() with null bounds means unbounded dimension" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
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
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
    }, .row_major);
    defer arr.deinit();

    var sliced = arr.slice(&[_][2]?isize{
        .{ 1, 3 },
        .{ null, null },
    });

    // Modify original array
    arr.set(&[_]isize{1, 0}, 999.0);

    // Slice should reflect the change (shares data)
    try testing.expectEqual(999.0, sliced.get(&[_]isize{0, 0}));
}

test "ndarray: slice() of 3D array extracts sub-tensor [2,3,4] → [1,3,4]" {
    const allocator = testing.allocator;
    var data: [24]i32 = undefined;
    for (0..24) |i| data[i] = @intCast(i);
    var arr = try NDArray(i32, 3).fromSlice(allocator, &[_]usize{2, 3, 4}, data[0..], .row_major);
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
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{3, 4}, &[_]f64{
        1,  2,  3,  4,
        5,  6,  7,  8,
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
    try testing.expectEqual(7.0, sliced.get(&[_]isize{0, 0}));
    try testing.expectEqual(8.0, sliced.get(&[_]isize{0, 1}));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{1, 0}));
    try testing.expectEqual(12.0, sliced.get(&[_]isize{1, 1}));
}

test "ndarray: slice() rejects out-of-bounds ranges" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
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
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set [-1, -1] (last element)
    arr.set(&[_]isize{-1, -1}, 42.0);

    // Verify via positive index
    try testing.expectEqual(42.0, arr.get(&[_]isize{2, 3}));

    // Set [-2, -3] (second-to-last row, third-to-last col)
    arr.set(&[_]isize{-2, -3}, 88.0);

    // Verify via positive index
    try testing.expectEqual(88.0, arr.get(&[_]isize{1, 1}));
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
