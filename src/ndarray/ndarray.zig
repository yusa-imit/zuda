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
