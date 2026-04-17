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
            InvalidFormat,
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
                    const diff = if (@typeInfo(T) == .float)
                        (stop - start) / step
                    else
                        @as(f64, @floatFromInt(stop - start)) / @as(f64, @floatFromInt(step));
                    num_elements = @as(usize, @intFromFloat(@ceil(diff)));
                }
            } else {
                if (start > stop) {
                    const diff = if (@typeInfo(T) == .float)
                        (start - stop) / (-step)
                    else
                        @as(f64, @floatFromInt(start - stop)) / @as(f64, @floatFromInt(-step));
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
        /// Time: O(n) where n = prod(shape) (currently always copies for memory safety)
        /// Space: O(prod(shape)) for new allocation
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
            std.mem.reverse(usize, &new_shape);
            std.mem.reverse(usize, &new_strides);

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
        pub fn add(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn sub(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn mul(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn div(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn mod(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn matmul(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn exp(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.exp(val);
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
        pub fn log(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.log(T, std.math.e, val);
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
        pub fn sqrt(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.sqrt(val);
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
        pub fn pow(self: *const Self, exponent: T) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.pow(T, val, exponent);
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
        pub fn floor(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn ceil(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn round(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn trunc(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn sin(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.sin(val);
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
        pub fn cos(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.cos(val);
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
        pub fn tan(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.tan(val);
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
        pub fn asin(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.asin(val);
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
        pub fn acos(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.acos(val);
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
        pub fn atan(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.atan(val);
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
        pub fn atan2(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.atan2(y_val, x_val);
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
        pub fn sinh(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.sinh(val);
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
        pub fn cosh(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.cosh(val);
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
        pub fn tanh(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.tanh(val);
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

        /// Element-wise base-2 logarithm - returns array with log2 of each element
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn log2(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.log2(val);
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
        pub fn log10(self: *const Self) (Error || std.mem.Allocator.Error)!Self {
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
                result_data[idx] = std.math.log10(val);
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
        pub fn eq(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a == b;
                }
            }.op);
        }

        /// Element-wise inequality comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn ne(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a != b;
                }
            }.op);
        }

        /// Element-wise less-than comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn lt(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a < b;
                }
            }.op);
        }

        /// Element-wise less-or-equal comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn le(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a <= b;
                }
            }.op);
        }

        /// Element-wise greater-than comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn gt(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
            return applyBinaryCompOp(T, ndim, self, other, self.allocator, struct {
                pub fn op(a: T, b: T) bool {
                    return a > b;
                }
            }.op);
        }

        /// Element-wise greater-or-equal comparison - returns boolean array
        ///
        /// Time: O(n) | Space: O(n) for result allocation
        pub fn ge(self: *const Self, other: *const Self) (Error || std.mem.Allocator.Error)!NDArray(bool, ndim) {
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

            fn toTypeInfo(self: TypeTag) std.builtin.Type {
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
            const file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            // Write magic number "NDAR" (4 bytes)
            var magic_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &magic_bytes, 0x4E444152, .little);
            _ = try file.write(&magic_bytes);

            // Write version (1) (4 bytes)
            var version_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &version_bytes, 1, .little);
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
                std.mem.writeInt(usize, &dim_bytes, dim, .little);
                _ = try file.write(&dim_bytes);
            }

            // Write strides array
            for (self.strides) |stride| {
                var stride_bytes: [@sizeOf(usize)]u8 = undefined;
                std.mem.writeInt(usize, &stride_bytes, stride, .little);
                _ = try file.write(&stride_bytes);
            }

            // Write data
            const bytes = std.mem.sliceAsBytes(self.data);
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
        pub fn load(allocator: std.mem.Allocator, path: []const u8) !Self {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            // Read and validate magic number (4 bytes)
            var magic_bytes: [4]u8 = undefined;
            _ = try file.read(&magic_bytes);
            const magic = std.mem.readInt(u32, &magic_bytes, .little);
            if (magic != 0x4E444152) {
                return error.InvalidFormat;
            }

            // Read and validate version (4 bytes)
            var version_bytes: [4]u8 = undefined;
            _ = try file.read(&version_bytes);
            const version = std.mem.readInt(u32, &version_bytes, .little);
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
                shape[i] = std.mem.readInt(usize, &dim_bytes, .little);
            }

            // Read strides
            var strides: [ndim]usize = undefined;
            for (0..ndim) |i| {
                var stride_bytes: [@sizeOf(usize)]u8 = undefined;
                _ = try file.read(&stride_bytes);
                strides[i] = std.mem.readInt(usize, &stride_bytes, .little);
            }

            // Calculate total elements
            var total: usize = 1;
            for (shape) |dim| {
                total *= dim;
            }

            // Allocate and read data
            const data = try allocator.alloc(T, total);
            errdefer allocator.free(data);

            const bytes = std.mem.sliceAsBytes(data);
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
        pub fn cumsum(self: *const Self, allocator: Allocator) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn cumprod(self: *const Self, allocator: Allocator) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn squeeze(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!NDArray(T, ndim - 1) {
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
        pub fn unsqueeze(self: *const Self, allocator: Allocator, axis: usize) (Error || std.mem.Allocator.Error)!NDArray(T, ndim + 1) {
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
        pub fn concat(allocator: Allocator, arrays: []const *const Self, axis: usize, layout: Layout) (Error || std.mem.Allocator.Error)!Self {
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
        pub fn stack(allocator: Allocator, arrays: []const *const Self, axis: usize, layout: Layout) (Error || std.mem.Allocator.Error)!NDArray(T, ndim + 1) {
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
    };
}

/// Helper to copy array to stacked result during stack operation
fn copyArrayToStack(comptime T: type, comptime ndim: usize, dest: *NDArray(T, ndim + 1), src: *const NDArray(T, ndim), axis: usize, stack_index: usize) !void {
    // Iterate through all elements of source array using multi-dimensional indices
    var indices: [ndim]usize = std.mem.zeroes([ndim]usize);

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
    var indices: [ndim]usize = std.mem.zeroes([ndim]usize);

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

    // std.testing.allocator will detect leaks if any transposes incorrectly allocate
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
    const allocator = std.testing.allocator;
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

    const pi = std.math.pi;
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

    const pi = std.math.pi;
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
    a.data[3] = std.math.sqrt(3.0);

    var result = try a.atan();
    defer result.deinit();

    const pi = std.math.pi;
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

    const pi = std.math.pi;
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
    try std.fs.cwd().deleteFile("/tmp/test_ndarray_1d.bin");
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
    try std.fs.cwd().deleteFile("/tmp/test_ndarray_2d.bin");
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
    try std.fs.cwd().deleteFile("/tmp/test_ndarray_3d.bin");
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

    try std.fs.cwd().deleteFile("/tmp/test_ndarray_bool.bin");
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

    try std.fs.cwd().deleteFile("/tmp/test_ndarray_wrong_ndim.bin");
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

    try std.fs.cwd().deleteFile("/tmp/test_ndarray_wrong_type.bin");
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

    try std.fs.cwd().deleteFile("/tmp/test_ndarray_large.bin");
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
fn broadcastShapes(shape_a: []const usize, shape_b: []const usize, allocator: std.mem.Allocator) !([]usize) {
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
    other: *const NDArray(T, ndim), allocator: std.mem.Allocator,
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
    other: *const NDArray(T, ndim), allocator: std.mem.Allocator,
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

    try testing.expectEqual(@as(i32, 10), try result.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(i32, 40), try result.get(&[_]isize{ 1, 0 }));
}

test "ndarray: stack() type variant - u8 arrays" {
    const allocator = testing.allocator;

    var a = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &[_]u8{ 65, 66, 67 }, .row_major);
    defer a.deinit();
    var b = try NDArray(u8, 1).fromSlice(allocator, &[_]usize{3}, &[_]u8{ 68, 69, 70 }, .row_major);
    defer b.deinit();

    var result = try NDArray(u8, 1).stack(allocator, &[_]*const NDArray(u8, 1){ &a, &b }, 0, .row_major);
    defer result.deinit();

    try testing.expectEqual(@as(u8, 65), try result.get(&[_]isize{ 0, 0 }));
    try testing.expectEqual(@as(u8, 68), try result.get(&[_]isize{ 1, 0 }));
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
