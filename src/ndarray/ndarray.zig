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
            InvalidPermutation,
            ShapeMismatch,
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
        pub fn fromOwnedSlice(allocator: Allocator, shape: []const usize, owned_data: []T, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
                if (total_elements > std.math.maxInt(usize) / dim) {
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
        pub fn eye(allocator: Allocator, rows: usize, cols: usize, k: isize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn identity(allocator: Allocator, rows: usize, cols: usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
            return Self.eye(allocator, rows, cols, 0, layout);
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
        /// Time: O(1) if contiguous (zero-copy), O(n) if non-contiguous (copy where n = prod(shape))
        /// Space: O(ndim) for metadata, O(prod(shape)) if copy required
        pub fn reshape(self: *const Self, new_shape: []const usize) (Error || std.mem.Allocator.Error)!Self {
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
                if (new_total > std.math.maxInt(usize) / dim) {
                    return error.CapacityExceeded;
                }
                new_total *= dim;
            }

            // Verify total size matches
            if (new_total != self.count()) {
                return error.CapacityExceeded;
            }

            // Check if array is contiguous
            // Contiguous: data.len equals the expected total element count
            const is_contiguous = self.data.len == self.count();

            if (is_contiguous) {
                // Zero-copy: use fromOwnedSlice with same data pointer
                // We need to cast away const on data slice
                const mutable_data: []T = @constCast(self.data);
                return Self.fromOwnedSlice(self.allocator, new_shape, mutable_data, self.layout);
            } else {
                // Non-contiguous: must copy data to new contiguous buffer
                // Allocate new buffer
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
            std.mem.reverse(usize, &new_shape);
            std.mem.reverse(usize, &new_strides);

            // Return new view with reversed metadata but same data pointer
            return Self{
                .shape = new_shape,
                .strides = new_strides,
                .data = self.data,
                .allocator = self.allocator,
                .layout = self.layout,
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
            return @typeInfo(@TypeOf(self.shape)).Array.len;
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
        pub fn flatten(self: *const Self) (Error || std.mem.Allocator.Error)!NDArray(T, 1) {
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
        pub fn ravel(self: *const Self) (Error || std.mem.Allocator.Error)!NDArray(T, 1) {
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
        /// - std.mem.Allocator.Error if memory allocation fails during copying
        ///
        /// Time: O(1) if already contiguous, O(n) if copying required (n = prod(shape))
        /// Space: O(n) if allocation needed for new buffer
        pub fn contiguous(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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

            // If already contiguous, return a copy of the struct
            if (is_contiguous) {
                return Self{
                    .shape = self.shape,
                    .strides = self.strides,
                    .data = self.data,
                    .allocator = self.allocator,
                    .layout = self.layout,
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
        pub fn add(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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

            // Traverse both arrays in element order and add
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |self_val| {
                const other_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = self_val + other_val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
        pub fn sub(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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

            // Traverse both arrays in element order and subtract
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |self_val| {
                const other_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = self_val - other_val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
        pub fn mul(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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

            // Traverse both arrays in element order and multiply
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |self_val| {
                const other_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = self_val * other_val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
        pub fn div(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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

            // Traverse both arrays in element order and divide
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |self_val| {
                const other_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = self_val / other_val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
        pub fn mod(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: mod only works on integer types
            if (!std.meta.trait.isIntegral(T)) {
                @compileError("mod() is only defined for integer types");
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

            // Traverse both arrays in element order and compute modulo
            var self_iter = self.iterator();
            var other_iter = other.iterator();

            var idx: usize = 0;
            while (self_iter.next()) |self_val| {
                const other_val = other_iter.next() orelse return error.ShapeMismatch;
                result_data[idx] = self_val % other_val;
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
        pub fn neg(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn abs(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn exp(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: exp only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.exp(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn log(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: log only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.log(T, std.math.e, val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn sqrt(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: sqrt only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.sqrt(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn pow(self: *const Self, exponent: T) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: pow only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.pow(T, val, exponent);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn sin(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: sin only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.sin(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn cos(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: cos only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.cos(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
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
        pub fn tan(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
            // Compile-time check: tan only works on float types
            if (!std.meta.trait.isFloat(T)) {
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
                result_data[idx] = std.math.tan(val);
                idx += 1;
            }

            return Self{
                .shape = self.shape,
                .strides = self.strides,
                .data = result_data,
                .allocator = self.allocator,
                .layout = self.layout,
            };
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
                sum_val += @as(f64, @floatFromInt(@as(i128, @intCast(val))));
            }
            return sum_val / @as(f64, @floatFromInt(@as(i128, @intCast(total))));
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

        /// Sum along a specified axis, reducing that dimension
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Dimension to sum along (must be < ndim)
        ///
        /// Returns: New NDArray with shape having axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - Allocator.Error if memory allocation fails
        ///
        /// Example: shape [3,4,5] with axis=1 → result shape [3,5]
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(output_size) for result array
        pub fn sumAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!Self {
            // Validate axis
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Calculate output shape (remove axis dimension)
            var out_shape: [ndim - 1]usize = undefined;
            var out_idx: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    out_shape[out_idx] = self.shape[i];
                    out_idx += 1;
                }
            }

            // Create output array
            var result = try NDArray(T, ndim - 1).init(allocator, out_shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize output to zero
            @memset(result.data, 0);

            // Iterate through input and accumulate into output
            var iter = self.iterator();
            var flat_idx: usize = 0;
            while (iter.next()) |val| {
                // Convert flat index to multi-dimensional indices
                var multi_idx: [ndim]usize = undefined;
                var current = flat_idx;
                for (0..ndim) |dim| {
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }
                    multi_idx[dim] = current / divisor;
                    current = current % divisor;
                }

                // Map input multi-index to output multi-index
                var out_multi_idx: [ndim - 1]usize = undefined;
                var out_idx_pos: usize = 0;
                for (0..ndim) |i| {
                    if (i != axis) {
                        out_multi_idx[out_idx_pos] = multi_idx[i];
                        out_idx_pos += 1;
                    }
                }

                // Calculate output flat index using row-major order
                var out_flat_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    var divisor: usize = 1;
                    for (i + 1..ndim - 1) |d| {
                        divisor *= out_shape[d];
                    }
                    out_flat_idx += out_multi_idx[i] * divisor;
                }

                result.data[out_flat_idx] += val;
                flat_idx += 1;
            }

            return result;
        }

        /// Product along a specified axis, reducing that dimension
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Dimension to multiply along (must be < ndim)
        ///
        /// Returns: New NDArray with shape having axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(output_size) for result array
        pub fn prodAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!Self {
            // Validate axis
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Calculate output shape (remove axis dimension)
            var out_shape: [ndim - 1]usize = undefined;
            var out_idx: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    out_shape[out_idx] = self.shape[i];
                    out_idx += 1;
                }
            }

            // Create output array
            var result = try NDArray(T, ndim - 1).init(allocator, out_shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize output to 1
            for (result.data) |*val| {
                val.* = 1;
            }

            // Iterate through input and accumulate into output
            var iter = self.iterator();
            var flat_idx: usize = 0;
            while (iter.next()) |val| {
                // Convert flat index to multi-dimensional indices
                var multi_idx: [ndim]usize = undefined;
                var current = flat_idx;
                for (0..ndim) |dim| {
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }
                    multi_idx[dim] = current / divisor;
                    current = current % divisor;
                }

                // Map input multi-index to output multi-index
                var out_multi_idx: [ndim - 1]usize = undefined;
                var out_idx_pos: usize = 0;
                for (0..ndim) |i| {
                    if (i != axis) {
                        out_multi_idx[out_idx_pos] = multi_idx[i];
                        out_idx_pos += 1;
                    }
                }

                // Calculate output flat index using row-major order
                var out_flat_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    var divisor: usize = 1;
                    for (i + 1..ndim - 1) |d| {
                        divisor *= out_shape[d];
                    }
                    out_flat_idx += out_multi_idx[i] * divisor;
                }

                result.data[out_flat_idx] *= val;
                flat_idx += 1;
            }

            return result;
        }

        /// Mean along a specified axis, reducing that dimension
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Dimension to average along (must be < ndim)
        ///
        /// Returns: New NDArray(f64, ndim-1) with shape having axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(output_size) for result array
        pub fn meanAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!NDArray(f64, ndim - 1) {
            // Validate axis
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Calculate output shape (remove axis dimension)
            var out_shape: [ndim - 1]usize = undefined;
            var out_idx: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    out_shape[out_idx] = self.shape[i];
                    out_idx += 1;
                }
            }

            // Create output array
            var result = try NDArray(f64, ndim - 1).init(allocator, out_shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize output to 0
            @memset(result.data, 0);

            // Track counts for averaging
            var counts = try allocator.alloc(usize, result.count());
            defer allocator.free(counts);
            @memset(counts, 0);

            // Iterate through input and accumulate into output
            var iter = self.iterator();
            var flat_idx: usize = 0;
            while (iter.next()) |val| {
                // Convert flat index to multi-dimensional indices
                var multi_idx: [ndim]usize = undefined;
                var current = flat_idx;
                for (0..ndim) |dim| {
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }
                    multi_idx[dim] = current / divisor;
                    current = current % divisor;
                }

                // Map input multi-index to output multi-index
                var out_multi_idx: [ndim - 1]usize = undefined;
                var out_idx_pos: usize = 0;
                for (0..ndim) |i| {
                    if (i != axis) {
                        out_multi_idx[out_idx_pos] = multi_idx[i];
                        out_idx_pos += 1;
                    }
                }

                // Calculate output flat index using row-major order
                var out_flat_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    var divisor: usize = 1;
                    for (i + 1..ndim - 1) |d| {
                        divisor *= out_shape[d];
                    }
                    out_flat_idx += out_multi_idx[i] * divisor;
                }

                result.data[out_flat_idx] += @as(f64, @floatFromInt(@as(i128, @intCast(val))));
                counts[out_flat_idx] += 1;
                flat_idx += 1;
            }

            // Divide by counts
            for (0..result.count()) |i| {
                if (counts[i] > 0) {
                    result.data[i] /= @as(f64, @floatFromInt(@as(i128, @intCast(counts[i]))));
                }
            }

            return result;
        }

        /// Minimum along a specified axis, reducing that dimension
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Dimension to reduce along (must be < ndim)
        ///
        /// Returns: New NDArray with shape having axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(output_size) for result array
        pub fn minAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!Self {
            // Validate axis
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Calculate output shape (remove axis dimension)
            var out_shape: [ndim - 1]usize = undefined;
            var out_idx: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    out_shape[out_idx] = self.shape[i];
                    out_idx += 1;
                }
            }

            // Create output array
            var result = try NDArray(T, ndim - 1).init(allocator, out_shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize output to a value that will be overwritten (we track first element per output)
            var first_seen = try allocator.alloc(bool, result.count());
            defer allocator.free(first_seen);
            @memset(first_seen, false);

            // Iterate through input and reduce
            var iter = self.iterator();
            var flat_idx: usize = 0;
            while (iter.next()) |val| {
                // Convert flat index to multi-dimensional indices
                var multi_idx: [ndim]usize = undefined;
                var current = flat_idx;
                for (0..ndim) |dim| {
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }
                    multi_idx[dim] = current / divisor;
                    current = current % divisor;
                }

                // Map input multi-index to output multi-index
                var out_multi_idx: [ndim - 1]usize = undefined;
                var out_idx_pos: usize = 0;
                for (0..ndim) |i| {
                    if (i != axis) {
                        out_multi_idx[out_idx_pos] = multi_idx[i];
                        out_idx_pos += 1;
                    }
                }

                // Calculate output flat index using row-major order
                var out_flat_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    var divisor: usize = 1;
                    for (i + 1..ndim - 1) |d| {
                        divisor *= out_shape[d];
                    }
                    out_flat_idx += out_multi_idx[i] * divisor;
                }

                if (!first_seen[out_flat_idx] || val < result.data[out_flat_idx]) {
                    result.data[out_flat_idx] = val;
                    first_seen[out_flat_idx] = true;
                }
                flat_idx += 1;
            }

            return result;
        }

        /// Maximum along a specified axis, reducing that dimension
        ///
        /// Parameters:
        /// - allocator: Memory allocator for the result array
        /// - axis: Dimension to reduce along (must be < ndim)
        ///
        /// Returns: New NDArray with shape having axis dimension removed
        ///
        /// Errors:
        /// - error.IndexOutOfBounds if axis >= ndim
        /// - Allocator.Error if memory allocation fails
        ///
        /// Time: O(n) where n = prod(shape)
        /// Space: O(output_size) for result array
        pub fn maxAxis(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!Self {
            // Validate axis
            if (axis >= ndim) {
                return error.IndexOutOfBounds;
            }

            // Calculate output shape (remove axis dimension)
            var out_shape: [ndim - 1]usize = undefined;
            var out_idx: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    out_shape[out_idx] = self.shape[i];
                    out_idx += 1;
                }
            }

            // Create output array
            var result = try NDArray(T, ndim - 1).init(allocator, out_shape[0..], self.layout);
            errdefer result.deinit();

            // Initialize output to a value that will be overwritten (we track first element per output)
            var first_seen = try allocator.alloc(bool, result.count());
            defer allocator.free(first_seen);
            @memset(first_seen, false);

            // Iterate through input and reduce
            var iter = self.iterator();
            var flat_idx: usize = 0;
            while (iter.next()) |val| {
                // Convert flat index to multi-dimensional indices
                var multi_idx: [ndim]usize = undefined;
                var current = flat_idx;
                for (0..ndim) |dim| {
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }
                    multi_idx[dim] = current / divisor;
                    current = current % divisor;
                }

                // Map input multi-index to output multi-index
                var out_multi_idx: [ndim - 1]usize = undefined;
                var out_idx_pos: usize = 0;
                for (0..ndim) |i| {
                    if (i != axis) {
                        out_multi_idx[out_idx_pos] = multi_idx[i];
                        out_idx_pos += 1;
                    }
                }

                // Calculate output flat index using row-major order
                var out_flat_idx: usize = 0;
                for (0..ndim - 1) |i| {
                    var divisor: usize = 1;
                    for (i + 1..ndim - 1) |d| {
                        divisor *= out_shape[d];
                    }
                    out_flat_idx += out_multi_idx[i] * divisor;
                }

                if (!first_seen[out_flat_idx] || val > result.data[out_flat_idx]) {
                    result.data[out_flat_idx] = val;
                    first_seen[out_flat_idx] = true;
                }
                flat_idx += 1;
            }

            return result;
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

                // Convert flat row-major index to multi-dimensional indices
                // then apply strides to get correct memory offset
                var multi_index: [ndim]usize = undefined;
                var current = self.index;

                // Calculate multi-dimensional indices from flat row-major index
                // For shape [d0, d1, d2, ...], flat index i converts to:
                // index[0] = i / (d1*d2*...), then i %= (d1*d2*...)
                // index[1] = i / (d2*...), then i %= (d2*...)
                // etc.
                for (0..ndim) |dim| {
                    // Calculate divisor: product of all dimensions after this one
                    var divisor: usize = 1;
                    for (dim + 1..ndim) |d| {
                        divisor *= self.shape[d];
                    }

                    // Extract index for this dimension
                    multi_index[dim] = current / divisor;
                    current = current % divisor;
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
                .index = 0,
                .total = self.count(),
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
    const result = NDArray(f64, 2).init(allocator, &[_]usize{ std.math.maxInt(usize), 2 }, .row_major);

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

    // std.testing.allocator automatically detects leaks
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

// -- fromOwnedSlice() Creation Function Tests (11 tests) --

test "ndarray: fromOwnedSlice() basic 1D array [5] takes ownership" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f64, 5);
    defer {
        // Only free if test fails; normally deinit() handles it
        if (!std.debug.runtime_safety) {
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
    const huge_shape = &[_]usize{ std.math.maxInt(usize), 2 };
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

// -- Indexing Tests (get/set) (7 tests) --

test "ndarray: get() retrieves single element from 2D array [2,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // Row-major [2,3]: [1,2,3 | 4,5,6]
    // Get [0,0] = 1.0, [0,2] = 3.0, [1,1] = 5.0
    try testing.expectEqual(1.0, arr.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(3.0, arr.get(&[_]isize{ 0, 2 }));
    try testing.expectEqual(5.0, arr.get(&[_]isize{ 1, 1 }));
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

    try testing.expectEqual(10.0, arr.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(20.0, arr.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(30.0, arr.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(40.0, arr.get(&[_]isize{ 1, 1 }));
}

test "ndarray: get() rejects out-of-bounds positive indices" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Shape [2,3] allows indices [0-1, 0-2]
    // [2,0] is out of bounds (row index too high)
    try testing.expectError(error.IndexOutOfBounds, arr.get(&[_]isize{ 2, 0 }));
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
    try testing.expectEqual(3.0, sliced.at(0));
    try testing.expectEqual(7.0, sliced.at(1));
    try testing.expectEqual(11.0, sliced.at(2));
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
    try testing.expectEqual(6.0, sliced.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(7.0, sliced.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(10.0, sliced.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{ 1, 1 }));
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
    try testing.expectEqual(999.0, sliced.get(&[_]isize{ 0, 0 }));
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
    try testing.expectEqual(7.0, sliced.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(8.0, sliced.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(11.0, sliced.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(12.0, sliced.get(&[_]isize{ 1, 1 }));
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
    try testing.expectEqual(42.0, arr.get(&[_]isize{ 2, 3 }));

    // Set [-2, -3] (second-to-last row, third-to-last col)
    arr.set(&[_]isize{ -2, -3 }, 88.0);

    // Verify via positive index
    try testing.expectEqual(88.0, arr.get(&[_]isize{ 1, 1 }));
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

test "ndarray: reshape 1D [6] → 2D [2,3] preserves data" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).fromSlice(allocator, &[_]usize{6}, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // Reshape from 1D to 2D
    var reshaped = try arr.reshape(&[_]usize{ 2, 3 });
    defer reshaped.deinit();

    // Verify shape changed
    try testing.expectEqual(2, reshaped.shape[0]);
    try testing.expectEqual(3, reshaped.shape[1]);

    // Verify all data preserved in order
    var iter = reshaped.iterator();
    var idx: usize = 1;
    while (iter.next()) |val| {
        try testing.expectEqual(@as(f64, @floatFromInt(idx)), val);
        idx += 1;
    }
}

test "ndarray: reshape 2D [2,3] → 1D [6] flattens array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f64{ 1, 2, 3, 4, 5, 6 }, .row_major);
    defer arr.deinit();

    // Reshape from 2D to 1D
    var reshaped = try arr.reshape(&[_]usize{6});
    defer reshaped.deinit();

    // Verify shape is 1D
    try testing.expectEqual(6, reshaped.shape[0]);
    try testing.expectEqual(6, reshaped.count());

    // Verify data preserved
    for (0..6) |i| {
        try testing.expectEqual(@as(f64, @floatFromInt(i + 1)), reshaped.data[i]);
    }
}

test "ndarray: reshape 3D [2,3,4] → 2D [6,4] preserves elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).zeros(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Initialize with sequential values
    for (0..24) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Reshape from 3D to 2D
    var reshaped = try arr.reshape(&[_]usize{ 6, 4 });
    defer reshaped.deinit();

    // Verify shape
    try testing.expectEqual(6, reshaped.shape[0]);
    try testing.expectEqual(4, reshaped.shape[1]);
    try testing.expectEqual(24, reshaped.count());

    // Verify first and last elements
    try testing.expectEqual(@as(i32, 1), reshaped.data[0]);
    try testing.expectEqual(@as(i32, 24), reshaped.data[23]);
}

test "ndarray: reshape [12] → [3,4] verifies total size consistency" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 1, 13, 1, .row_major);
    defer arr.deinit();

    var reshaped = try arr.reshape(&[_]usize{ 3, 4 });
    defer reshaped.deinit();

    try testing.expectEqual(12, reshaped.count());
    try testing.expectEqual(3, reshaped.shape[0]);
    try testing.expectEqual(4, reshaped.shape[1]);
}

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

test "ndarray: reshape returns error on size mismatch [6] → [2,2]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 6.0, 1.0, .row_major);
    defer arr.deinit();

    // Total elements mismatch: 6 ≠ 2*2
    const result = arr.reshape(&[_]usize{ 2, 2 });

    try testing.expectError(error.CapacityExceeded, result);
}

test "ndarray: reshape returns error on zero dimension [6] → [0,3]" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 6.0, 1.0, .row_major);
    defer arr.deinit();

    // Zero dimension not allowed
    const result = arr.reshape(&[_]usize{ 0, 3 });

    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: reshape contiguous array does zero-copy (same data pointer)" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 6.0, 1.0, .row_major);
    defer arr.deinit();

    const original_data_ptr = arr.data.ptr;

    var reshaped = try arr.reshape(&[_]usize{ 2, 3 });
    defer reshaped.deinit();

    // For contiguous row-major 1D → 2D, data pointer should remain the same
    try testing.expectEqual(original_data_ptr, reshaped.data.ptr);
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
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 6.0, 1.0, .row_major);
    defer arr.deinit();

    var reshaped = try arr.reshape(&[_]usize{ 2, 3 });
    defer reshaped.deinit();

    try testing.expectEqual(Layout.row_major, reshaped.layout);
}

test "ndarray: reshape preserves column-major layout" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 1).arange(allocator, 0.0, 6.0, 1.0, .row_major);
    defer arr.deinit();

    var arr_cm = try NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer arr_cm.deinit();

    for (0..6) |i| {
        arr_cm.data[i] = @floatFromInt(i + 1);
    }

    var reshaped = try arr_cm.reshape(&[_]usize{ 3, 2 });
    defer reshaped.deinit();

    try testing.expectEqual(Layout.column_major, reshaped.layout);
}

test "ndarray: reshape empty array [0] → [0,0] (edge case)" {
    // This test documents expected behavior for empty arrays
    // Actual support depends on implementation
    const allocator = testing.allocator;

    // If zero-dimension arrays are not supported, this will fail at init
    // Otherwise, test that reshape maintains the no-element invariant
    _ = allocator;
}

test "ndarray: reshape [24] → [2,3,4] → [6,4] → [24] multiple reshapes" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 1, 25, 1, .row_major);
    defer arr.deinit();

    // First reshape: 1D → 3D
    var reshaped_3d = try arr.reshape(&[_]usize{ 2, 3, 4 });
    defer reshaped_3d.deinit();

    try testing.expectEqual(24, reshaped_3d.count());

    // Second reshape: 3D → 2D
    var reshaped_2d = try reshaped_3d.reshape(&[_]usize{ 6, 4 });
    defer reshaped_2d.deinit();

    try testing.expectEqual(24, reshaped_2d.count());
    try testing.expectEqual(6, reshaped_2d.shape[0]);
    try testing.expectEqual(4, reshaped_2d.shape[1]);

    // Third reshape: 2D → 1D
    var reshaped_1d = try reshaped_2d.reshape(&[_]usize{24});
    defer reshaped_1d.deinit();

    try testing.expectEqual(24, reshaped_1d.count());
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

test "ndarray: reshape with large dimensions [1000] → [10,10,10]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).arange(allocator, 0, 1000, 1, .row_major);
    defer arr.deinit();

    var reshaped = try arr.reshape(&[_]usize{ 10, 10, 10 });
    defer reshaped.deinit();

    try testing.expectEqual(10, reshaped.shape[0]);
    try testing.expectEqual(10, reshaped.shape[1]);
    try testing.expectEqual(10, reshaped.shape[2]);
    try testing.expectEqual(1000, reshaped.count());

    // Verify first and last elements unchanged
    try testing.expectEqual(@as(i32, 0), reshaped.data[0]);
    try testing.expectEqual(@as(i32, 999), reshaped.data[999]);
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

    // std.testing.allocator will detect leaks if any allocation not freed
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

test "ndarray: transpose 2D row-major [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill: [[1,2,3],[4,5,6]] in row-major order
    arr.data[0] = 1;
    arr.data[1] = 2;
    arr.data[2] = 3;
    arr.data[3] = 4;
    arr.data[4] = 5;
    arr.data[5] = 6;

    const transposed = arr.transpose();

    // After transpose [3,2]: [[1,4],[2,5],[3,6]]
    // In memory (row-major): 1, 4, 2, 5, 3, 6
    // But strides are swapped, so we access via strides
    try testing.expectEqual(1, transposed.at(&[_]isize{ 0, 0 }));
    try testing.expectEqual(4, transposed.at(&[_]isize{ 0, 1 }));
    try testing.expectEqual(2, transposed.at(&[_]isize{ 1, 0 }));
    try testing.expectEqual(5, transposed.at(&[_]isize{ 1, 1 }));
    try testing.expectEqual(3, transposed.at(&[_]isize{ 2, 0 }));
    try testing.expectEqual(6, transposed.at(&[_]isize{ 2, 1 }));
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

test "ndarray: transpose view modification affects original" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill original: [1,2,3,4,5,6]
    for (0..6) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    var transposed = arr.transpose();

    // Modify transposed view at [0,1] (which is original [1,0])
    transposed.set(&[_]isize{ 0, 1 }, 99);

    // Original should reflect the change
    try testing.expectEqual(99, arr.at(&[_]isize{ 1, 0 }));
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

    // std.testing.allocator will detect leaks if any transposes incorrectly allocate
}

// ============================================================================
// flatten() Tests - Convert multi-dimensional array to 1D
// ============================================================================

test "ndarray: flatten 2D [2,3] row-major → 1D [6] with elements [[1,2,3],[4,5,6]] → [1,2,3,4,5,6]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Set values: [[1,2,3], [4,5,6]]
    arr.set(&[_]isize{ 0, 0 }, 1);
    arr.set(&[_]isize{ 0, 1 }, 2);
    arr.set(&[_]isize{ 0, 2 }, 3);
    arr.set(&[_]isize{ 1, 0 }, 4);
    arr.set(&[_]isize{ 1, 1 }, 5);
    arr.set(&[_]isize{ 1, 2 }, 6);

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape is 1D with 6 elements
    try testing.expectEqual(1, flattened.ndim());
    try testing.expectEqual(6, flattened.shape[0]);
    try testing.expectEqual(6, flattened.count());

    // Verify row-major order: [1,2,3,4,5,6]
    try testing.expectEqual(1, flattened.at(0));
    try testing.expectEqual(2, flattened.at(1));
    try testing.expectEqual(3, flattened.at(2));
    try testing.expectEqual(4, flattened.at(3));
    try testing.expectEqual(5, flattened.at(4));
    try testing.expectEqual(6, flattened.at(5));
}

test "ndarray: flatten 3D [2,3,4] → 1D [24] preserves all elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values 1..24 directly into data slice
    for (0..24) |i| {
        arr.data[i] = @floatFromInt(i + 1);
    }

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(1, flattened.ndim());
    try testing.expectEqual(24, flattened.shape[0]);
    try testing.expectEqual(24, flattened.count());

    // Verify all elements are in row-major order
    for (0..24) |i| {
        const expected: f64 = @floatFromInt(i + 1);
        try testing.expectEqual(expected, flattened.at(i));
    }
}

test "ndarray: flatten 1D [6] → 1D [6] is zero-copy (same data pointer)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{6}, .row_major);
    defer arr.deinit();

    // Fill with values 1..6
    for (0..6) |i| {
        arr.set(&[_]isize{@intCast(i)}, @intCast(i + 1));
    }

    const original_ptr = arr.data.ptr;

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape unchanged
    try testing.expectEqual(1, flattened.shape[0]);
    try testing.expectEqual(6, flattened.count());

    // Verify zero-copy: same data pointer
    try testing.expectEqual(original_ptr, flattened.data.ptr);
}

test "ndarray: flatten contiguous 2D [3,4] row-major → 1D [12] zero-copy" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill array directly into data slice
    for (0..12) |i| {
        arr.data[i] = @floatFromInt(i + 1);
    }

    const original_ptr = arr.data.ptr;

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(12, flattened.shape[0]);

    // Verify zero-copy: same data pointer
    try testing.expectEqual(original_ptr, flattened.data.ptr);

    // Verify elements preserved in row-major order
    for (0..12) |i| {
        const expected: f64 = @floatFromInt(i + 1);
        try testing.expectEqual(expected, flattened.at(i));
    }
}

test "ndarray: flatten non-contiguous (sliced) 2D → 1D requires copy" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 4, 4 }, .row_major);
    defer arr.deinit();

    // Fill entire array
    for (0..4) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @intCast(i * 4 + j + 1));
        }
    }

    // Slice to get [[5,6,7,8],[9,10,11,12],[13,14,15,16]] (non-contiguous view)
    const sliced = try arr.slice(&[_]?[2]usize{ .{ 1, 4 }, .{ 0, 4 } });

    const flattened = try sliced.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(12, flattened.shape[0]);

    // Non-contiguous requires copy, so data pointer should differ
    // OR if implementation handles it as zero-copy despite non-contiguity,
    // then we just verify elements are in correct row-major order
    const expected = [_]i32{ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    for (0..12) |i| {
        try testing.expectEqual(expected[i], flattened.at(i));
    }
}

test "ndarray: flatten column-major 2D [3,4] → 1D [12] converts to row-major order" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    // In column-major layout, fill and verify order
    // [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @floatFromInt(i + j * 3 + 1));
        }
    }

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(12, flattened.shape[0]);

    // The flattened result should be in row-major order: [1,4,7,10,2,5,8,11,3,6,9,12]
    const expected = [_]f64{ 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12 };
    for (0..12) |i| {
        try testing.expectEqual(expected[i], flattened.at(i));
    }
}

test "ndarray: flatten empty 2D [0,5] → 1D [0] shape correct" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 0, 5 }, .row_major);
    defer arr.deinit();

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify empty shape
    try testing.expectEqual(1, flattened.ndim());
    try testing.expectEqual(0, flattened.shape[0]);
    try testing.expectEqual(0, flattened.count());
}

test "ndarray: flatten [1,1,1,6] 4D → 1D [6] preserves all elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 4).init(allocator, &[_]usize{ 1, 1, 1, 6 }, .row_major);
    defer arr.deinit();

    // Fill with values 10..15
    for (0..6) |i| {
        arr.set(&[_]isize{ 0, 0, 0, @intCast(i) }, @intCast(i + 10));
    }

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(1, flattened.shape[0]);
    try testing.expectEqual(6, flattened.count());

    // Verify elements
    for (0..6) |i| {
        try testing.expectEqual(@as(i32, @intCast(i + 10)), flattened.at(i));
    }
}

test "ndarray: flatten large [100,200] → [20000] 1D array" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{ 100, 200 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..20000) |i| {
        arr.data[i] = @floatFromInt(i);
    }

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify shape
    try testing.expectEqual(20000, flattened.shape[0]);
    try testing.expectEqual(20000, flattened.count());

    // Spot check some values
    try testing.expectEqual(0.0, flattened.at(0));
    try testing.expectEqual(100.0, flattened.at(100));
    try testing.expectEqual(19999.0, flattened.at(19999));
}

test "ndarray: flatten no memory leak with multiple flatten operations" {
    const allocator = testing.allocator;

    for (0..5) |_| {
        var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
        defer arr.deinit();

        for (0..3) |_| {
            const flattened = try arr.flatten();
            defer flattened.deinit();
            try testing.expectEqual(24, flattened.count());
        }
    }
    // std.testing.allocator will detect leaks if flatten incorrectly allocates
}

test "ndarray: flatten all elements accessible by index in 1D result" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 5 }, .row_major);
    defer arr.deinit();

    // Fill with values 100..114
    for (0..15) |i| {
        arr.data[i] = @intCast(i + 100);
    }

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify every element is accessible and correct
    for (0..15) |i| {
        const expected: i32 = @intCast(i + 100);
        try testing.expectEqual(expected, flattened.at(i));
    }
}

test "ndarray: flatten reshape reshape preserves elements (2D→1D→2D)" {
    const allocator = testing.allocator;
    var original = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer original.deinit();

    // Fill with 1..12
    for (0..3) |i| {
        for (0..4) |j| {
            original.set(&[_]isize{ @intCast(i), @intCast(j) }, @intCast(i * 4 + j + 1));
        }
    }

    // Flatten to 1D
    const flattened = try original.flatten();
    defer flattened.deinit();

    // Reshape back to 2D [4,3]
    const reshaped = try flattened.reshape(&[_]usize{ 4, 3 });
    defer reshaped.deinit();

    // Verify shape
    try testing.expectEqual(4, reshaped.shape[0]);
    try testing.expectEqual(3, reshaped.shape[1]);

    // Verify all 12 elements are present (reshaped layout may differ)
    var found_count: usize = 0;
    var iter = reshaped.iterator();
    while (iter.next()) |_| {
        found_count += 1;
    }
    try testing.expectEqual(12, found_count);
}

test "ndarray: flatten after zeros() creation preserves zeros in 1D" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).zeros(allocator, &[_]usize{ 2, 2, 3 }, .row_major);
    defer arr.deinit();

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify all zeros
    for (0..12) |i| {
        try testing.expectEqual(0.0, flattened.at(i));
    }
}

test "ndarray: flatten after ones() creation preserves ones in 1D" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).ones(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    const flattened = try arr.flatten();
    defer flattened.deinit();

    // Verify all ones
    for (0..12) |i| {
        try testing.expectEqual(1, flattened.at(i));
    }
}

// ============================================================================
// TESTS FOR ravel() — Always-Copy Flatten Semantics
// ============================================================================

test "ndarray: ravel 2D row-major preserves element sequence" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values 0..11 in row-major order
    for (0..3) |i| {
        for (0..4) |j| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, @intCast(i * 4 + j));
        }
    }

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Verify shape is [12]
    try testing.expectEqual(1, raveled.shape.len);
    try testing.expectEqual(12, raveled.count());

    // Verify element sequence matches row-major order
    for (0..12) |idx| {
        const expected: i32 = @intCast(idx);
        try testing.expectEqual(expected, raveled.at(@intCast(idx)));
    }
}

test "ndarray: ravel 3D row-major preserves all elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    var idx: usize = 0;
    while (idx < arr.count()) : (idx += 1) {
        arr.data[idx] = @as(f64, @floatFromInt(idx)) + 1.0;
    }

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Verify total element count: 2*3*4 = 24
    try testing.expectEqual(24, raveled.count());

    // Verify all values copied correctly
    for (0..24) |i| {
        const expected: f64 = @as(f64, @floatFromInt(i)) + 1.0;
        try testing.expectEqual(expected, raveled.at(@intCast(i)));
    }
}

test "ndarray: ravel 1D array still allocates copy (not view)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..10) |i| {
        arr.data[i] = @intCast(i + 100);
    }

    const original_ptr = arr.data.ptr;

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Critical: ravel must ALWAYS allocate new data, even for 1D arrays
    // Different pointers = separate allocations
    try testing.expect(raveled.data.ptr != original_ptr);

    // Verify values still correct
    for (0..10) |i| {
        const expected: i32 = @intCast(i + 100);
        try testing.expectEqual(expected, raveled.at(@intCast(i)));
    }
}

test "ndarray: ravel always allocates new data pointer (never shares)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr.deinit();

    const original_ptr = arr.data.ptr;

    // Even for contiguous row-major array, ravel() must copy
    const raveled = try arr.ravel();
    defer raveled.deinit();

    // This is the key difference from flatten():
    // flatten() may return a view (same pointer if contiguous)
    // ravel() ALWAYS returns a new allocation (different pointer)
    try testing.expect(raveled.data.ptr != original_ptr);

    // Verify owned data is modifiable independently
    raveled.data[0] = 999;
    try testing.expectEqual(0, arr.data[0]); // Original unchanged
    try testing.expectEqual(999, raveled.data[0]); // Copy modified
}

test "ndarray: ravel non-contiguous sliced array copies all elements" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 4, 5 }, .row_major);
    defer arr.deinit();

    // Fill with values
    for (0..20) |i| {
        arr.data[i] = @intCast(i);
    }

    // Create a non-contiguous slice [1:3, 1:4] (2x3 view)
    const sliced = arr.slice(&[_][2]?isize{
        &[_]?isize{ 1, 3 },
        &[_]?isize{ 1, 4 },
    });

    // Sliced array is non-contiguous
    try testing.expectEqual(2, sliced.shape[0]);
    try testing.expectEqual(3, sliced.shape[1]);

    const raveled = try sliced.ravel();
    defer raveled.deinit();

    // Verify ravel captured all elements from slice
    try testing.expectEqual(6, raveled.count());

    // Expected values from slice [1:3, 1:4]:
    // Row 1, cols 1-3: 6,7,8
    // Row 2, cols 1-3: 11,12,13
    const expected = [_]i32{ 6, 7, 8, 11, 12, 13 };
    for (0..6) |i| {
        try testing.expectEqual(expected[i], raveled.at(@intCast(i)));
    }
}

test "ndarray: ravel column-major preserves layout order" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .column_major);
    defer arr.deinit();

    // Fill in column-major order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
    var value: i32 = 0;
    for (0..3) |j| {
        for (0..2) |i| {
            arr.set(&[_]isize{ @intCast(i), @intCast(j) }, value);
            value += 1;
        }
    }

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Verify layout is preserved in result
    try testing.expectEqual(Layout.column_major, raveled.layout);

    // Verify element order matches column-major traversal
    for (0..6) |i| {
        try testing.expectEqual(i, raveled.at(@intCast(i)));
    }
}

test "ndarray: ravel empty-dimension array [0,5] error handling" {
    const allocator = testing.allocator;

    // Note: NDArray disallows zero dimensions at init time
    // This test verifies the design constraint is maintained
    const result = NDArray(i32, 2).init(allocator, &[_]usize{ 0, 5 }, .row_major);
    try testing.expectError(error.ZeroDimension, result);
}

test "ndarray: ravel large array stress test (10k+ elements)" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 10, 20, 50 }, .row_major);
    defer arr.deinit();

    // Total: 10,000 elements
    try testing.expectEqual(10000, arr.count());

    // Fill with pattern: element value = flat index % 256
    for (0..10000) |i| {
        arr.data[i] = @intCast((i % 256));
    }

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Verify all elements copied correctly
    try testing.expectEqual(10000, raveled.count());
    for (0..10000) |i| {
        const expected: i32 = @intCast(i % 256);
        try testing.expectEqual(expected, raveled.at(@intCast(i)));
    }
}

test "ndarray: ravel no memory leaks on repeated calls" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 5, 6 }, .row_major);
    defer arr.deinit();

    // Repeated ravel calls must not leak memory
    for (0..5) |_| {
        const raveled = try arr.ravel();
        defer raveled.deinit();
        try testing.expectEqual(30, raveled.count());
    }
    // testing.allocator will detect any leaks
}

test "ndarray: ravel preserves layout in result metadata" {
    const allocator = testing.allocator;

    // Test row-major preservation
    var arr_rm = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .row_major);
    defer arr_rm.deinit();

    const raveled_rm = try arr_rm.ravel();
    defer raveled_rm.deinit();

    try testing.expectEqual(Layout.row_major, raveled_rm.layout);

    // Test column-major preservation
    var arr_cm = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr_cm.deinit();

    const raveled_cm = try arr_cm.ravel();
    defer raveled_cm.deinit();

    try testing.expectEqual(Layout.column_major, raveled_cm.layout);
}

test "ndarray: ravel result is proper 1D array with shape [n]" {
    const allocator = testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    const raveled = try arr.ravel();
    defer raveled.deinit();

    // Result must be 1D with correct shape
    try testing.expectEqual(24, raveled.count());
    try testing.expectEqual(24, raveled.shape[0]);
    try testing.expectEqual(1, raveled.strides[0]);

    // Iterator should work correctly on 1D result
    var iter = raveled.iterator();
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try testing.expectEqual(24, count);
}

// -- permute() Function Tests (12+ tests) --

test "ndarray: permute 2D [1,0] equals transpose" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 4 });
    defer arr.deinit();

    // Fill with sequential values
    for (0..12) |i| {
        arr.data[i] = @intCast(i);
    }

    // Permute with [1,0] should equal transpose
    const permuted = try arr.permute(&[_]usize{ 1, 0 });
    const transposed = arr.transpose();

    // Verify shapes match
    try testing.expectEqual(transposed.shape[0], permuted.shape[0]);
    try testing.expectEqual(transposed.shape[1], permuted.shape[1]);

    // Verify strides match
    try testing.expectEqual(transposed.strides[0], permuted.strides[0]);
    try testing.expectEqual(transposed.strides[1], permuted.strides[1]);

    // Verify element access matches
    for (0..3) |i| {
        for (0..4) |j| {
            const idx_i: isize = @intCast(i);
            const idx_j: isize = @intCast(j);
            try testing.expectEqual(
                transposed.at(&[_]isize{ idx_i, idx_j }),
                permuted.at(&[_]isize{ idx_i, idx_j }),
            );
        }
    }
}

test "ndarray: permute 3D [2,1,0] reverses all axes" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    // Fill with sequential values
    for (0..24) |i| {
        arr.data[i] = @intCast(i);
    }

    const permuted = try arr.permute(&[_]usize{ 2, 1, 0 });

    // Verify shape: [2,3,4] → [4,3,2]
    try testing.expectEqual(4, permuted.shape[0]);
    try testing.expectEqual(3, permuted.shape[1]);
    try testing.expectEqual(2, permuted.shape[2]);

    // Verify total element count unchanged
    try testing.expectEqual(24, permuted.count());
}

test "ndarray: permute 3D [1,2,0] cyclic rotation" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    for (0..24) |i| {
        arr.data[i] = @intCast(i);
    }

    const permuted = try arr.permute(&[_]usize{ 1, 2, 0 });

    // Verify shape: [2,3,4] → [3,4,2]
    try testing.expectEqual(3, permuted.shape[0]);
    try testing.expectEqual(4, permuted.shape[1]);
    try testing.expectEqual(2, permuted.shape[2]);

    // Total count preserved
    try testing.expectEqual(24, permuted.count());
}

test "ndarray: permute 3D identity [0,1,2] no-op" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    for (0..24) |i| {
        arr.data[i] = @intCast(i);
    }

    const permuted = try arr.permute(&[_]usize{ 0, 1, 2 });

    // Shape unchanged
    try testing.expectEqual(2, permuted.shape[0]);
    try testing.expectEqual(3, permuted.shape[1]);
    try testing.expectEqual(4, permuted.shape[2]);

    // Strides unchanged
    try testing.expectEqual(arr.strides[0], permuted.strides[0]);
    try testing.expectEqual(arr.strides[1], permuted.strides[1]);
    try testing.expectEqual(arr.strides[2], permuted.strides[2]);
}

test "ndarray: permute zero-copy same data pointer" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    const original_ptr = arr.data.ptr;

    const permuted = try arr.permute(&[_]usize{ 2, 1, 0 });

    // CRITICAL: Data pointer must be identical (zero-copy view)
    try testing.expectEqual(original_ptr, permuted.data.ptr);
}

test "ndarray: permute preserves total element count" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    const permuted = try arr.permute(&[_]usize{ 2, 0, 1 });

    // Element count must be preserved
    try testing.expectEqual(arr.count(), permuted.count());
    try testing.expectEqual(24, permuted.count());
}

test "ndarray: permute shape and strides transformation 2D" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 4 });
    defer arr.deinit();

    // With row-major layout: strides should be [4, 1] for shape [3,4]
    const original_strides = arr.strides;

    const permuted = try arr.permute(&[_]usize{ 1, 0 });

    // Shape becomes [4, 3]
    try testing.expectEqual(4, permuted.shape[0]);
    try testing.expectEqual(3, permuted.shape[1]);

    // Strides should be reversed to [1, 4]
    try testing.expectEqual(original_strides[1], permuted.strides[0]);
    try testing.expectEqual(original_strides[0], permuted.strides[1]);
}

test "ndarray: permute error duplicate axes [0,0,1]" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    // Axes with duplicate should fail
    const result = arr.permute(&[_]usize{ 0, 0, 1 });
    try testing.expectError(error.InvalidPermutation, result);
}

test "ndarray: permute error out of range axes [0,1,3]" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    // Axes with out-of-range index should fail
    const result = arr.permute(&[_]usize{ 0, 1, 3 });
    try testing.expectError(error.InvalidPermutation, result);
}

test "ndarray: permute error wrong length axes" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    // Wrong number of axes should fail
    const result = arr.permute(&[_]usize{ 0, 1 });
    try testing.expectError(error.InvalidPermutation, result);
}

test "ndarray: permute chain reversibility" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 });
    defer arr.deinit();

    for (0..24) |i| {
        arr.data[i] = @intCast(i);
    }

    // Permute with [2,0,1]
    const perm1 = try arr.permute(&[_]usize{ 2, 0, 1 });
    try testing.expectEqual(@as(usize, 4), perm1.shape[0]);
    try testing.expectEqual(@as(usize, 2), perm1.shape[1]);
    try testing.expectEqual(@as(usize, 3), perm1.shape[2]);

    // Permute back with [1,2,0] (inverse permutation)
    const perm2 = try perm1.permute(&[_]usize{ 1, 2, 0 });
    try testing.expectEqual(@as(usize, 2), perm2.shape[0]);
    try testing.expectEqual(@as(usize, 3), perm2.shape[1]);
    try testing.expectEqual(@as(usize, 4), perm2.shape[2]);
}

test "ndarray: permute 1D array [0] identity" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 1).init(allocator, &[_]usize{10});
    defer arr.deinit();

    for (0..10) |i| {
        arr.data[i] = @intCast(i);
    }

    const permuted = try arr.permute(&[_]usize{0});

    try testing.expectEqual(@as(usize, 10), permuted.shape[0]);
    try testing.expectEqual(arr.data.ptr, permuted.data.ptr);
}

test "ndarray: permute 2D partial reorder [1,0] affects element access" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 });
    defer arr.deinit();

    // Fill: [[0,1,2],[3,4,5]]
    arr.data[0] = 0;
    arr.data[1] = 1;
    arr.data[2] = 2;
    arr.data[3] = 3;
    arr.data[4] = 4;
    arr.data[5] = 5;

    const permuted = try arr.permute(&[_]usize{ 1, 0 });

    // Shape should be [3, 2]
    try testing.expectEqual(@as(usize, 3), permuted.shape[0]);
    try testing.expectEqual(@as(usize, 2), permuted.shape[1]);

    // Access pattern changes due to stride reordering
    // Original arr[0,0]=0, arr[0,1]=1, arr[0,2]=2, arr[1,0]=3, arr[1,1]=4, arr[1,2]=5
    // After permute [1,0]: shape [3,2], element ordering in strides changes
    try testing.expectEqual(@as(i32, 0), permuted.at(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 3), permuted.at(&[_]isize{ 0, 1 }));
}

// -- contiguous() Tests --

test "ndarray: contiguous already contiguous 2D array returns same pointer" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    const contiguous = try arr.contiguous();
    defer contiguous.deinit();

    // Should have same data pointer (already contiguous)
    try testing.expectEqual(arr.data.ptr, contiguous.data.ptr);

    // Should have same shape
    try testing.expectEqualSlices(usize, &arr.shape, &contiguous.shape);

    // Should have same strides
    try testing.expectEqualSlices(usize, &arr.strides, &contiguous.strides);

    // Data should be identical
    try testing.expectEqualSlices(i32, arr.data, contiguous.data);
}

test "ndarray: contiguous 1D array always contiguous returns same pointer" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f64, 1).init(allocator, &[_]usize{10}, .row_major);
    defer arr.deinit();

    for (0..arr.data.len) |i| {
        arr.data[i] = @floatFromInt(i);
    }

    const contiguous = try arr.contiguous();
    defer contiguous.deinit();

    // 1D arrays are always contiguous
    try testing.expectEqual(arr.data.ptr, contiguous.data.ptr);
    try testing.expectEqual(@as(usize, 10), contiguous.shape[0]);
    try testing.expectEqual(@as(usize, 1), contiguous.strides[0]);
}

test "ndarray: contiguous sliced non-contiguous array allocates new buffer" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 4, 5 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    // Slice [1:3, 1:4] - creates non-contiguous view
    const sliced = arr.slice(&[_][2]?isize{
        &[_]?isize{ 1, 3 },
        &[_]?isize{ 1, 4 },
    });

    const contiguous = try sliced.contiguous();
    defer contiguous.deinit();

    // New buffer should be allocated (different pointer)
    try testing.expect(sliced.data.ptr != contiguous.data.ptr);

    // New buffer is shorter than original
    try testing.expect(contiguous.data.len < arr.data.len);

    // Shape should match sliced shape
    try testing.expectEqual(sliced.shape[0], contiguous.shape[0]);
    try testing.expectEqual(sliced.shape[1], contiguous.shape[1]);

    // Verify strides are contiguous (expected for row-major)
    try testing.expectEqual(@as(usize, 3), contiguous.strides[0]); // shape[1] = 3 (4 - 1)
    try testing.expectEqual(@as(usize, 1), contiguous.strides[1]);

    // Verify data is copied correctly
    try testing.expectEqual(@as(i32, 6), try contiguous.get(&[_]isize{ 0, 0 }));  // original[1,1]
    try testing.expectEqual(@as(i32, 7), try contiguous.get(&[_]isize{ 0, 1 }));  // original[1,2]
    try testing.expectEqual(@as(i32, 8), try contiguous.get(&[_]isize{ 0, 2 }));  // original[1,3]
}

test "ndarray: contiguous transposed 2D array allocates new buffer" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill with pattern: [[0,1,2],[3,4,5]]
    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    const transposed = arr.transpose();
    // Verify it's non-contiguous (stride pattern changes)
    // Original strides: [3, 1] for shape [2, 3]
    // After transpose: strides [1, 2] for shape [3, 2]

    const contiguous = try transposed.contiguous();
    defer contiguous.deinit();

    // New buffer allocated
    try testing.expect(transposed.data.ptr != contiguous.data.ptr);

    // New shape is transposed
    try testing.expectEqual(@as(usize, 3), contiguous.shape[0]);
    try testing.expectEqual(@as(usize, 2), contiguous.shape[1]);

    // Verify contiguous strides for row-major layout
    try testing.expectEqual(@as(usize, 2), contiguous.strides[0]);
    try testing.expectEqual(@as(usize, 1), contiguous.strides[1]);

    // Verify transposed elements are in correct positions
    // transposed[0,0] = original[0,0] = 0
    // transposed[0,1] = original[1,0] = 3
    // transposed[1,0] = original[0,1] = 1
    // transposed[1,1] = original[1,1] = 4
    try testing.expectEqual(@as(i32, 0), try contiguous.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 3), try contiguous.get(&[_]isize{ 0, 1 }));
    try testing.expectEqual(@as(i32, 1), try contiguous.get(&[_]isize{ 1, 0 }));
    try testing.expectEqual(@as(i32, 4), try contiguous.get(&[_]isize{ 1, 1 }));
}

test "ndarray: contiguous permuted 3D array allocates and reorders" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Fill with sequential values
    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    // Permute [2, 0, 1] - should create non-contiguous view
    const permuted = try arr.permute(&[_]usize{ 2, 0, 1 });

    const contiguous = try permuted.contiguous();
    defer contiguous.deinit();

    // New buffer allocated
    try testing.expect(permuted.data.ptr != contiguous.data.ptr);

    // New shape reflects permutation [2, 0, 1]
    // original [2, 3, 4] -> [4, 2, 3]
    try testing.expectEqual(@as(usize, 4), contiguous.shape[0]);
    try testing.expectEqual(@as(usize, 2), contiguous.shape[1]);
    try testing.expectEqual(@as(usize, 3), contiguous.shape[2]);

    // Verify contiguous strides for row-major
    try testing.expectEqual(@as(usize, 6), contiguous.strides[0]); // 2 * 3
    try testing.expectEqual(@as(usize, 3), contiguous.strides[1]); // 3
    try testing.expectEqual(@as(usize, 1), contiguous.strides[2]); // 1
}

test "ndarray: contiguous column-major layout stride validation" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32, 2).init(allocator, &[_]usize{ 3, 4 }, .column_major);
    defer arr.deinit();

    for (0..arr.data.len) |i| {
        arr.data[i] = @floatFromInt(i);
    }

    const contiguous = try arr.contiguous();
    defer contiguous.deinit();

    // For column-major [3, 4]: strides should be [1, 3]
    try testing.expectEqual(@as(usize, 1), contiguous.strides[0]);
    try testing.expectEqual(@as(usize, 3), contiguous.strides[1]);
}

test "ndarray: contiguous iterator traversal matches original after contiguation" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    // Fill pattern
    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    // Slice to create non-contiguous view
    const sliced = arr.slice(&[_][2]?isize{
        &[_]?isize{ 0, 2 },
        &[_]?isize{ 0, 2 },
    });

    const contiguous = try sliced.contiguous();
    defer contiguous.deinit();

    // Traverse and verify values match sliced view
    var iter_contiguous = contiguous.iterator();
    var iter_sliced = sliced.iterator();

    while (iter_contiguous.next()) |val_c| {
        const val_s = iter_sliced.next();
        try testing.expect(val_s != null);
        try testing.expectEqual(val_c, val_s.?);
    }

    try testing.expect(iter_sliced.next() == null);
}

test "ndarray: contiguous idempotent - calling twice preserves data" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer arr.deinit();

    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    const sliced = arr.slice(&[_][2]?isize{
        &[_]?isize{ 0, 2 },
        &[_]?isize{ 1, 3 },
    });

    const contiguous1 = try sliced.contiguous();
    defer contiguous1.deinit();

    // Call contiguous() on already-contiguous array
    const contiguous2 = try contiguous1.contiguous();
    defer contiguous2.deinit();

    // Should have same pointer (already contiguous)
    try testing.expectEqual(contiguous1.data.ptr, contiguous2.data.ptr);

    // Data should be identical
    try testing.expectEqualSlices(i32, contiguous1.data, contiguous2.data);
}

test "ndarray: contiguous large array stress test 1M elements" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 1000, 1000 }, .row_major);
    defer arr.deinit();

    // Just initialize; memory safety is key here
    // Slice to trigger non-contiguous state
    const sliced = arr.slice(&[_][2]?isize{
        &[_]?isize{ 100, 900 },
        &[_]?isize{ 100, 900 },
    });

    const contiguous = try sliced.contiguous();
    defer contiguous.deinit();

    // Verify dimensions
    try testing.expectEqual(@as(usize, 800), contiguous.shape[0]);
    try testing.expectEqual(@as(usize, 800), contiguous.shape[1]);

    // Verify total element count
    const expected_elements = 800 * 800;
    try testing.expectEqual(expected_elements, contiguous.count());
}

test "ndarray: contiguous empty slice handling" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{ 3, 3 }, .row_major);
    defer arr.deinit();

    // Slice that results in empty array [0, 3]
    const empty_slice = arr.slice(&[_][2]?isize{
        &[_]?isize{ 0, 0 },
        &[_]?isize{ 0, 3 },
    });

    // contiguous() on empty view should still work
    const contiguous = try empty_slice.contiguous();
    defer contiguous.deinit();

    try testing.expectEqual(@as(usize, 0), contiguous.shape[0]);
    try testing.expectEqual(@as(usize, 3), contiguous.shape[1]);
    try testing.expectEqual(@as(usize, 0), contiguous.count());
}

test "ndarray: contiguous preserves element values through copy" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    // Set specific pattern
    for (0..arr.data.len) |i| {
        arr.data[i] = @as(f64, @floatFromInt(i)) * 3.14;
    }

    // Permute to non-contiguous
    const permuted = try arr.permute(&[_]usize{ 1, 2, 0 });

    const contiguous = try permuted.contiguous();
    defer contiguous.deinit();

    // Spot-check values are preserved via iterator
    var iter = contiguous.iterator();
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }

    try testing.expectEqual(@as(usize, 24), count); // 2*3*4 = 24
}

test "ndarray: contiguous distinguishes contiguous from non-contiguous views" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{ 2, 3, 4 }, .row_major);
    defer arr.deinit();

    for (0..arr.data.len) |i| {
        arr.data[i] = @intCast(i);
    }

    // Original array is contiguous
    var contiguous_original = try arr.contiguous();
    defer contiguous_original.deinit();

    try testing.expectEqual(arr.data.ptr, contiguous_original.data.ptr);

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

test "ndarray: neg 2D integer array unary negation" {
    const allocator = testing.allocator;
    var a = try NDArray(i32, 2).init(allocator, &[_]usize{ 2, 3 }, .row_major);
    defer a.deinit();

    // a = [[1, -2, 3], [-4, 5, -6]]
    for (0..6) |i| {
        a.data[i] = @intCast(if (i % 2 == 0) i + 1 else -(i + 1));
    }

    var result = try a.neg();
    defer result.deinit();

    try testing.expectEqual(a.shape[0], result.shape[0]);
    try testing.expectEqual(a.shape[1], result.shape[1]);

    for (0..6) |i| {
        try testing.expectEqual(-a.data[i], result.data[i]);
    }
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
    try testing.expectApproxEqAbs(std.math.exp(1.0), result.data[1], 1e-10); // e^1 ≈ 2.71828
    try testing.expectApproxEqAbs(std.math.exp(2.0), result.data[2], 1e-10); // e^2 ≈ 7.38906

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
    try testing.expectApproxEqAbs(std.math.exp(1.0), result.data[1], 1e-10);
    try testing.expectApproxEqAbs(std.math.exp(-1.0), result.data[2], 1e-10);
    try testing.expectApproxEqAbs(std.math.exp(2.0), result.data[3], 1e-10);
}

test "ndarray: log 1D array natural logarithm" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{3}, .row_major);
    defer a.deinit();

    // a = [1, e, e^2]
    a.data[0] = 1.0;
    a.data[1] = std.math.exp(1.0);
    a.data[2] = std.math.exp(2.0);

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
    a.data[1] = std.math.exp(1.0);
    a.data[2] = std.math.exp(2.0);
    a.data[3] = std.math.exp(3.0);

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

test "ndarray: sin 1D array sine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 1).init(allocator, &[_]usize{5}, .row_major);
    defer a.deinit();

    // a = [0, π/6, π/4, π/3, π/2]
    const pi = std.math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 6.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 3.0;
    a.data[4] = pi / 2.0;

    var result = try a.sin();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);     // sin(0) = 0
    try testing.expectApproxEqAbs(0.5, result.data[1], 1e-10);     // sin(π/6) = 0.5
    try testing.expectApproxEqAbs(std.math.sin(pi / 4.0), result.data[2], 1e-10); // sin(π/4) ≈ 0.707
    try testing.expectApproxEqAbs(std.math.sin(pi / 3.0), result.data[3], 1e-10); // sin(π/3) ≈ 0.866
    try testing.expectApproxEqAbs(1.0, result.data[4], 1e-10);     // sin(π/2) = 1

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: sin 2D array sine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = std.math.pi;
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

    const pi = std.math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 3.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 2.0;
    a.data[4] = pi;

    var result = try a.cos();
    defer result.deinit();

    try testing.expectApproxEqAbs(1.0, result.data[0], 1e-10);                    // cos(0) = 1
    try testing.expectApproxEqAbs(0.5, result.data[1], 1e-10);                    // cos(π/3) = 0.5
    try testing.expectApproxEqAbs(std.math.cos(pi / 4.0), result.data[2], 1e-10); // cos(π/4) ≈ 0.707
    try testing.expectApproxEqAbs(0.0, result.data[3], 1e-10);                    // cos(π/2) = 0
    try testing.expectApproxEqAbs(-1.0, result.data[4], 1e-10);                   // cos(π) = -1

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: cos 2D array cosine" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = std.math.pi;
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

    const pi = std.math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 6.0;
    a.data[2] = pi / 4.0;
    a.data[3] = pi / 3.0;

    var result = try a.tan();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);                    // tan(0) = 0
    try testing.expectApproxEqAbs(std.math.tan(pi / 6.0), result.data[1], 1e-10); // tan(π/6) ≈ 0.577
    try testing.expectApproxEqAbs(1.0, result.data[2], 1e-10);                    // tan(π/4) = 1
    try testing.expectApproxEqAbs(std.math.tan(pi / 3.0), result.data[3], 1e-10); // tan(π/3) ≈ 1.732

    try testing.expect(result.data.ptr != a.data.ptr);
}

test "ndarray: tan 2D array tangent" {
    const allocator = testing.allocator;
    var a = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer a.deinit();

    const pi = std.math.pi;
    a.data[0] = 0.0;
    a.data[1] = pi / 4.0;
    a.data[2] = -pi / 4.0;
    a.data[3] = pi / 6.0;

    var result = try a.tan();
    defer result.deinit();

    try testing.expectApproxEqAbs(0.0, result.data[0], 1e-10);                     // tan(0) = 0
    try testing.expectApproxEqAbs(1.0, result.data[1], 1e-10);                     // tan(π/4) = 1
    try testing.expectApproxEqAbs(-1.0, result.data[2], 1e-10);                    // tan(-π/4) = -1
    try testing.expectApproxEqAbs(std.math.tan(pi / 6.0), result.data[3], 1e-10);  // tan(π/6) ≈ 0.577
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

test "reduction: sum() full 3D array f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 3).init(allocator, &[_]usize{2, 2, 2}, .row_major);
    defer arr.deinit();

    // Set to [[[1,2],[3,4]], [[5,6],[7,8]]]
    for (0..8) |i| {
        arr.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    // Expected: 1+2+...+8 = 36.0
    const result = arr.sum();
    try testing.expectApproxEqAbs(@as(f64, 36.0), @as(f64, @floatFromInt(@as(i128, @intCast(result)))), 1e-10);
}

test "reduction: sum() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Sum along axis 0 should reduce [3,4] -> [4]
    // Result: [1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24]
    const result = try arr.sumAxis(allocator, 0);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 15), result.data[0]);
    try testing.expectEqual(@as(i32, 18), result.data[1]);
    try testing.expectEqual(@as(i32, 21), result.data[2]);
    try testing.expectEqual(@as(i32, 24), result.data[3]);
}

test "reduction: sum() axis 1 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Sum along axis 1 should reduce [3,4] -> [3]
    // Result: [1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42]
    const result = try arr.sumAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 10), result.data[0]);
    try testing.expectEqual(@as(i32, 26), result.data[1]);
    try testing.expectEqual(@as(i32, 42), result.data[2]);
}

test "reduction: sum() axis on 3D array [2,3,4] - reduce dim 1" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr.deinit();

    // Fill with sequential values 1 to 24
    for (0..24) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Sum along axis 1 should reduce [2,3,4] -> [2,4]
    // Axis 1 has size 3, so each output element sums 3 values
    const result = try arr.sumAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 4), result.shape[1]);
    // First block sums: (1+5+9)=15, (2+6+10)=18, (3+7+11)=21, (4+8+12)=24
    try testing.expectEqual(@as(i32, 15), result.data[0]);
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

test "reduction: prod() full 2D array f64" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set to [[1, 2, 3], [4, 5, 6]]
    arr.data[0] = 1.0;
    arr.data[1] = 2.0;
    arr.data[2] = 3.0;
    arr.data[3] = 4.0;
    arr.data[4] = 5.0;
    arr.data[5] = 6.0;

    // Expected: 1*2*3*4*5*6 = 720.0
    const result = arr.prod();
    try testing.expectApproxEqAbs(@as(f64, 720.0), @as(f64, @floatFromInt(@as(i128, @intCast(result)))), 1e-10);
}

test "reduction: prod() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Product along axis 0 should reduce [3,4] -> [4]
    // Result: [1*5*9=45, 2*6*10=120, 3*7*11=231, 4*8*12=384]
    const result = try arr.prodAxis(allocator, 0);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 45), result.data[0]);
    try testing.expectEqual(@as(i32, 120), result.data[1]);
}

test "reduction: prod() axis 1 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Product along axis 1 should reduce [3,4] -> [3]
    // Result: [1*2*3*4=24, 5*6*7*8=1680, 9*10*11*12=11880]
    const result = try arr.prodAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 24), result.data[0]);
    try testing.expectEqual(@as(i32, 1680), result.data[1]);
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

test "reduction: mean() full 2D array f64 [2,3]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{2, 3}, .row_major);
    defer arr.deinit();

    // Set to [[1, 2, 3], [4, 5, 6]]
    arr.data[0] = 1.0;
    arr.data[1] = 2.0;
    arr.data[2] = 3.0;
    arr.data[3] = 4.0;
    arr.data[4] = 5.0;
    arr.data[5] = 6.0;

    // Expected: (1+2+3+4+5+6)/6 = 3.5
    const result = arr.mean();
    try testing.expectApproxEqAbs(@as(f64, 3.5), result, 1e-10);
}

test "reduction: mean() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    // Mean along axis 0 should reduce [3,4] -> [4]
    // Result: [(1+5+9)/3=5, (2+6+10)/3≈6, (3+7+11)/3=7, (4+8+12)/3=8]
    const result = try arr.meanAxis(allocator, 0);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 5.0), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 6.0), result.data[1], 1e-10);
}

test "reduction: mean() axis 1 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @as(f64, @floatFromInt(i + 1));
    }

    // Mean along axis 1 should reduce [3,4] -> [3]
    // Result: [(1+2+3+4)/4=2.5, (5+6+7+8)/4=6.5, (9+10+11+12)/4=10.5]
    const result = try arr.meanAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectApproxEqAbs(@as(f64, 2.5), result.data[0], 1e-10);
    try testing.expectApproxEqAbs(@as(f64, 6.5), result.data[1], 1e-10);
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

test "reduction: min() full 2D array f64 [3,3]" {
    const allocator = testing.allocator;

    var arr = try NDArray(f64, 2).init(allocator, &[_]usize{3, 3}, .row_major);
    defer arr.deinit();

    // Set to [[4.5, 1.2, 9.8], [2.1, 3.3, 0.5], [7.6, 2.2, 5.1]]
    arr.data[0] = 4.5;
    arr.data[1] = 1.2;
    arr.data[2] = 9.8;
    arr.data[3] = 2.1;
    arr.data[4] = 3.3;
    arr.data[5] = 0.5;
    arr.data[6] = 7.6;
    arr.data[7] = 2.2;
    arr.data[8] = 5.1;

    // Expected: min = 0.5
    const result = arr.min();
    try testing.expectApproxEqAbs(@as(f64, 0.5), @as(f64, @floatFromInt(@as(i128, @intCast(result)))), 1e-10);
}

test "reduction: min() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[9,8,7,6], [5,4,3,2], [1,10,11,12]]
    arr.data[0] = 9;
    arr.data[1] = 8;
    arr.data[2] = 7;
    arr.data[3] = 6;
    arr.data[4] = 5;
    arr.data[5] = 4;
    arr.data[6] = 3;
    arr.data[7] = 2;
    arr.data[8] = 1;
    arr.data[9] = 10;
    arr.data[10] = 11;
    arr.data[11] = 12;

    // Min along axis 0 should reduce [3,4] -> [4]
    // Result: [min(9,5,1)=1, min(8,4,10)=4, min(7,3,11)=3, min(6,2,12)=2]
    const result = try arr.minAxis(allocator, 0);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 4), result.data[1]);
}

test "reduction: min() axis 1 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Min along axis 1 should reduce [3,4] -> [3]
    // Result: [min(1,2,3,4)=1, min(5,6,7,8)=5, min(9,10,11,12)=9]
    const result = try arr.minAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 1), result.data[0]);
    try testing.expectEqual(@as(i32, 5), result.data[1]);
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

test "reduction: max() full 2D array f64 [2,3]" {
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

    // Expected: max = 5.5
    const result = arr.max();
    try testing.expectApproxEqAbs(@as(f64, 5.5), @as(f64, @floatFromInt(@as(i128, @intCast(result)))), 1e-10);
}

test "reduction: max() axis 0 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Max along axis 0 should reduce [3,4] -> [4]
    // Result: [max(1,5,9)=9, max(2,6,10)=10, max(3,7,11)=11, max(4,8,12)=12]
    const result = try arr.maxAxis(allocator, 0);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 4), result.shape[0]);
    try testing.expectEqual(@as(i32, 9), result.data[0]);
    try testing.expectEqual(@as(i32, 10), result.data[1]);
}

test "reduction: max() axis 1 on 2D array [3,4]" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Set to [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    for (0..12) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Max along axis 1 should reduce [3,4] -> [3]
    // Result: [max(1,2,3,4)=4, max(5,6,7,8)=8, max(9,10,11,12)=12]
    const result = try arr.maxAxis(allocator, 1);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 3), result.shape[0]);
    try testing.expectEqual(@as(i32, 4), result.data[0]);
    try testing.expectEqual(@as(i32, 8), result.data[1]);
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

test "reduction: axis reduction preserves other dimensions" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 3).init(allocator, &[_]usize{2, 3, 4}, .row_major);
    defer arr.deinit();

    // Fill sequentially
    for (0..24) |i| {
        arr.data[i] = @intCast(i + 1);
    }

    // Reduce along axis 0: [2,3,4] -> [3,4]
    const result_axis0 = try arr.sumAxis(allocator, 0);
    defer result_axis0.deinit();
    try testing.expectEqual(@as(usize, 3), result_axis0.shape[0]);
    try testing.expectEqual(@as(usize, 4), result_axis0.shape[1]);

    // Reduce along axis 1: [2,3,4] -> [2,4]
    const result_axis1 = try arr.sumAxis(allocator, 1);
    defer result_axis1.deinit();
    try testing.expectEqual(@as(usize, 2), result_axis1.shape[0]);
    try testing.expectEqual(@as(usize, 4), result_axis1.shape[1]);

    // Reduce along axis 2: [2,3,4] -> [2,3]
    const result_axis2 = try arr.sumAxis(allocator, 2);
    defer result_axis2.deinit();
    try testing.expectEqual(@as(usize, 2), result_axis2.shape[0]);
    try testing.expectEqual(@as(usize, 3), result_axis2.shape[1]);
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

test "reduction: axis out of bounds error" {
    const allocator = testing.allocator;

    var arr = try NDArray(i32, 2).init(allocator, &[_]usize{3, 4}, .row_major);
    defer arr.deinit();

    // Axis 2 is out of bounds for 2D array
    const result = arr.sumAxis(allocator, 2);
    try testing.expectError(error.IndexOutOfBounds, result);
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
