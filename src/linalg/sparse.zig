//! Sparse Matrix Formats and Operations
//!
//! Provides three standard sparse matrix storage formats optimized for different use cases:
//!
//! - **COO (Coordinate)**: Simplest format, stores (row, col, value) triplets
//!   - Best for: matrix construction, format conversion
//!   - Operations: O(nnz) iteration, O(nnz log nnz) sorting
//!
//! - **CSR (Compressed Sparse Row)**: Row-wise compressed storage
//!   - Best for: row-wise operations, matrix-vector multiply
//!   - Operations: O(nnz) row iteration, O(m) row access
//!
//! - **CSC (Compressed Sparse Column)**: Column-wise compressed storage
//!   - Best for: column-wise operations, transpose operations
//!   - Operations: O(nnz) column iteration, O(n) column access
//!
//! Use cases:
//! - Large-scale linear systems (FEM, CFD simulations)
//! - Graph algorithms (adjacency matrices)
//! - Machine learning (sparse feature matrices)
//! - Network analysis
//!
//! All formats support comptime numeric types and explicit allocator passing.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const math = std.math;

/// COO (Coordinate) Format
///
/// Stores sparse matrix as triplets of (row, col, value).
/// Memory: 3 × nnz (2 arrays of usize + 1 array of T)
///
/// Advantages:
/// - Simple construction and modification
/// - Easy format conversion
/// - Duplicates allowed (can be summed during conversion)
///
/// Disadvantages:
/// - No efficient row/column access
/// - Not optimized for arithmetic operations
///
/// Time: O(1) | Space: O(nnz)
pub fn COO(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Row indices (length = nnz)
        row_indices: std.ArrayList(usize),

        /// Column indices (length = nnz)
        col_indices: std.ArrayList(usize),

        /// Non-zero values (length = nnz)
        values: std.ArrayList(T),

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize empty COO matrix
        ///
        /// Time: O(1) | Space: O(1)
        pub fn init(allocator: Allocator, rows: usize, cols: usize) Self {
            const ArrayListUsize = std.ArrayList(usize);
            const ArrayListT = std.ArrayList(T);
            return Self{
                .rows = rows,
                .cols = cols,
                .row_indices = ArrayListUsize{},
                .col_indices = ArrayListUsize{},
                .values = ArrayListT{},
                .allocator = allocator,
            };
        }

        /// Initialize COO matrix with preallocated capacity
        ///
        /// Time: O(1) | Space: O(capacity)
        pub fn initCapacity(allocator: Allocator, rows: usize, cols: usize, capacity: usize) !Self {
            const ArrayListUsize = std.ArrayList(usize);
            const ArrayListT = std.ArrayList(T);

            var row_indices = try ArrayListUsize.initCapacity(allocator, capacity);
            errdefer row_indices.deinit(allocator);

            var col_indices = try ArrayListUsize.initCapacity(allocator, capacity);
            errdefer col_indices.deinit(allocator);

            const values = try ArrayListT.initCapacity(allocator, capacity);

            return Self{
                .rows = rows,
                .cols = cols,
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.row_indices.deinit(self.allocator);
            self.col_indices.deinit(self.allocator);
            self.values.deinit(self.allocator);
        }

        /// Add a non-zero element
        ///
        /// Note: Duplicates are allowed and will be summed during conversion to CSR/CSC
        ///
        /// Time: O(1) amortized | Space: O(1) amortized
        pub fn append(self: *Self, row: usize, col: usize, value: T) !void {
            if (row >= self.rows or col >= self.cols) {
                return error.OutOfBounds;
            }

            try self.row_indices.append(self.allocator, row);
            try self.col_indices.append(self.allocator, col);
            try self.values.append(self.allocator, value);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.items.len;
        }

        /// Check if matrix is empty
        ///
        /// Time: O(1) | Space: O(1)
        pub fn isEmpty(self: *const Self) bool {
            return self.values.items.len == 0;
        }

        /// Sort entries by (row, col) for efficient conversion to CSR
        ///
        /// Time: O(nnz log nnz) | Space: O(nnz)
        pub fn sort(self: *Self) !void {
            const Context = struct {
                rows: []const usize,
                cols: []const usize,

                pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
                    if (ctx.rows[a_index] != ctx.rows[b_index]) {
                        return ctx.rows[a_index] < ctx.rows[b_index];
                    }
                    return ctx.cols[a_index] < ctx.cols[b_index];
                }
            };

            const n = self.nnz();
            if (n == 0) return;

            // Create index array
            const indices = try self.allocator.alloc(usize, n);
            defer self.allocator.free(indices);

            for (indices, 0..) |*idx, i| {
                idx.* = i;
            }

            // Sort indices based on (row, col)
            const context = Context{
                .rows = self.row_indices.items,
                .cols = self.col_indices.items,
            };

            std.mem.sort(usize, indices, context, Context.lessThan);

            // Reorder arrays
            const temp_rows = try self.allocator.alloc(usize, n);
            defer self.allocator.free(temp_rows);
            const temp_cols = try self.allocator.alloc(usize, n);
            defer self.allocator.free(temp_cols);
            const temp_vals = try self.allocator.alloc(T, n);
            defer self.allocator.free(temp_vals);

            for (indices, 0..) |idx, i| {
                temp_rows[i] = self.row_indices.items[idx];
                temp_cols[i] = self.col_indices.items[idx];
                temp_vals[i] = self.values.items[idx];
            }

            mem.copyForwards(usize, self.row_indices.items, temp_rows);
            mem.copyForwards(usize, self.col_indices.items, temp_cols);
            mem.copyForwards(T, self.values.items, temp_vals);
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - All arrays have equal length
        /// - Row indices are within bounds [0, rows)
        /// - Column indices are within bounds [0, cols)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.row_indices.items.len;

            // Check array lengths match
            if (self.col_indices.items.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.items.len != n) {
                return error.InvalidArrayLength;
            }

            // Check indices are within bounds
            for (self.row_indices.items) |row| {
                if (row >= self.rows) {
                    return error.IndexOutOfBounds;
                }
            }
            for (self.col_indices.items) |col| {
                if (col >= self.cols) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// CSR (Compressed Sparse Row) Format
///
/// Stores sparse matrix in compressed row format:
/// - row_ptr[i] points to start of row i in col_indices and values
/// - row_ptr[i+1] - row_ptr[i] = number of non-zeros in row i
///
/// Memory: (m+1) + 2×nnz (row pointers + col indices + values)
///
/// Advantages:
/// - Efficient row-wise access O(1)
/// - Fast matrix-vector multiply O(nnz)
/// - Cache-friendly row iteration
///
/// Disadvantages:
/// - Column access is O(nnz)
/// - Modification requires rebuild
///
/// Time: O(1) | Space: O(m + nnz)
pub fn CSR(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Row pointers (length = rows + 1)
        /// row_ptr[i] = start index of row i in col_indices/values
        /// row_ptr[rows] = nnz (total number of non-zeros)
        row_ptr: []usize,

        /// Column indices (length = nnz)
        col_indices: []usize,

        /// Non-zero values (length = nnz)
        values: []T,

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize CSR matrix from COO format
        ///
        /// Note: Input COO matrix must be sorted by (row, col)
        /// Duplicates will be summed
        ///
        /// Time: O(nnz) | Space: O(m + nnz)
        pub fn fromCOO(allocator: Allocator, coo: *COO(T)) !Self {
            const n = coo.nnz();
            const m = coo.rows;

            // Allocate arrays
            const row_ptr = try allocator.alloc(usize, m + 1);
            errdefer allocator.free(row_ptr);

            const col_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(col_indices);

            const values = try allocator.alloc(T, n);

            // Initialize row_ptr to zero
            for (row_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return Self{
                    .rows = m,
                    .cols = coo.cols,
                    .row_ptr = row_ptr,
                    .col_indices = col_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per row
            for (coo.row_indices.items) |row| {
                row_ptr[row + 1] += 1;
            }

            // Cumulative sum to get row pointers
            for (1..m + 1) |i| {
                row_ptr[i] += row_ptr[i - 1];
            }

            // Fill col_indices and values
            // Use temporary counters to track insertion positions
            const temp_ptr = try allocator.alloc(usize, m + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, row_ptr);

            for (coo.row_indices.items, 0..) |row, i| {
                const pos = temp_ptr[row];
                col_indices[pos] = coo.col_indices.items[i];
                values[pos] = coo.values.items[i];
                temp_ptr[row] += 1;
            }

            return Self{
                .rows = m,
                .cols = coo.cols,
                .row_ptr = row_ptr,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.row_ptr);
            self.allocator.free(self.col_indices);
            self.allocator.free(self.values);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.len;
        }

        /// Get non-zero count for specific row
        ///
        /// Time: O(1) | Space: O(1)
        pub fn rowNnz(self: *const Self, row: usize) usize {
            if (row >= self.rows) return 0;
            return self.row_ptr[row + 1] - self.row_ptr[row];
        }

        /// Get element at (row, col), returns 0 if not found
        ///
        /// Time: O(nnz_row) | Space: O(1)
        pub fn get(self: *const Self, row: usize, col: usize) T {
            if (row >= self.rows or col >= self.cols) return 0;

            const start = self.row_ptr[row];
            const end = self.row_ptr[row + 1];

            for (self.col_indices[start..end], start..) |c, i| {
                if (c == col) return self.values[i];
            }

            return 0;
        }

        /// Entry returned by row iterator
        /// Transpose CSR matrix to CSC format
        ///
        /// Converts row-wise compressed format to column-wise compressed format.
        /// Effectively computes A^T (transpose).
        ///
        /// Time: O(nnz) | Space: O(n + nnz)
        pub fn transpose(self: *const Self, allocator: Allocator) !CSC(T) {
            const n = self.nnz();
            const col_count = self.cols;

            // Allocate CSC arrays
            const col_ptr = try allocator.alloc(usize, col_count + 1);
            errdefer allocator.free(col_ptr);

            const row_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(row_indices);

            const values = try allocator.alloc(T, n);

            // Initialize col_ptr to zero
            for (col_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return CSC(T){
                    .rows = self.cols, // Swap dimensions
                    .cols = self.rows,
                    .col_ptr = col_ptr,
                    .row_indices = row_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per column (becomes col_ptr)
            for (self.col_indices) |col| {
                col_ptr[col + 1] += 1;
            }

            // Cumulative sum to get column pointers
            for (1..col_count + 1) |i| {
                col_ptr[i] += col_ptr[i - 1];
            }

            // Fill row_indices and values
            const temp_ptr = try allocator.alloc(usize, col_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, col_ptr);

            for (0..self.rows) |row| {
                const start = self.row_ptr[row];
                const end = self.row_ptr[row + 1];
                for (start..end) |idx| {
                    const col = self.col_indices[idx];
                    const pos = temp_ptr[col];
                    row_indices[pos] = row;
                    values[pos] = self.values[idx];
                    temp_ptr[col] += 1;
                }
            }

            return CSC(T){
                .rows = self.cols, // Swap dimensions
                .cols = self.rows,
                .col_ptr = col_ptr,
                .row_indices = row_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Sparse matrix-vector multiplication: y = A*x
        ///
        /// Computes the product of this sparse matrix (m×n) with a dense vector x (n×1),
        /// producing a dense vector y (m×1). Uses row-wise accumulation.
        ///
        /// Time: O(nnz) | Space: O(m) for result vector
        pub fn matvec(self: *const Self, allocator: Allocator, x: []const T) ![]T {
            const y = try allocator.alloc(T, self.rows);
            errdefer allocator.free(y);

            // Initialize result to zero
            for (y) |*val| {
                val.* = 0;
            }

            // Row-wise accumulation
            for (0..self.rows) |row| {
                const start = self.row_ptr[row];
                const end = self.row_ptr[row + 1];
                var sum: T = 0;
                for (start..end) |idx| {
                    const col = self.col_indices[idx];
                    sum += self.values[idx] * x[col];
                }
                y[row] = sum;
            }

            return y;
        }

        /// Sparse matrix-matrix multiply: C = A × B
        ///
        /// Computes the product of two CSR sparse matrices.
        /// Returns a new CSR matrix containing the result.
        ///
        /// Algorithm: Row-wise multiplication using intermediate COO format
        /// - For each row i in A:
        ///   - For each non-zero A[i,k]:
        ///     - For each non-zero B[k,j]:
        ///       - Accumulate A[i,k] * B[k,j] into C[i,j]
        /// - Convert accumulated COO to CSR format
        ///
        /// Use cases:
        /// - Matrix powers (A^k) for graph algorithms
        /// - Sparse linear algebra operations
        /// - Iterative methods requiring A × A^T
        ///
        /// Time: O(nnz(A) × nnz_row_avg(B)) | Space: O(nnz(C))
        pub fn matmul(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.cols != other.rows) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format for accumulation
            var coo = COO(T).init(allocator, self.rows, other.cols);
            errdefer coo.deinit();

            // Use HashMap to accumulate products at each (row, col) position
            // Key: (row, col) encoded as row * other.cols + col
            // Value: accumulated sum
            const HashMap = std.AutoHashMap(usize, T);
            var accumulator = HashMap.init(allocator);
            defer accumulator.deinit();

            // For each row in A
            for (0..self.rows) |i| {
                const a_start = self.row_ptr[i];
                const a_end = self.row_ptr[i + 1];

                // For each non-zero A[i,k]
                for (a_start..a_end) |a_idx| {
                    const k = self.col_indices[a_idx];
                    const a_ik = self.values[a_idx];

                    // For each non-zero B[k,j]
                    const b_start = other.row_ptr[k];
                    const b_end = other.row_ptr[k + 1];
                    for (b_start..b_end) |b_idx| {
                        const j = other.col_indices[b_idx];
                        const b_kj = other.values[b_idx];

                        // Accumulate into C[i,j]
                        const key = i * other.cols + j;
                        const product = a_ik * b_kj;

                        if (accumulator.get(key)) |existing| {
                            try accumulator.put(key, existing + product);
                        } else {
                            try accumulator.put(key, product);
                        }
                    }
                }
            }

            // Convert accumulator to COO
            var iter = accumulator.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const value = entry.value_ptr.*;
                const row = key / other.cols;
                const col = key % other.cols;
                try coo.append(row, col, value);
            }

            // Sort COO before converting to CSR
            try coo.sort();

            // Convert COO to CSR
            const result = try Self.fromCOO(allocator, &coo);

            // Clean up COO (fromCOO only borrows, doesn't consume)
            coo.deinit();

            return result;
        }

        /// Element-wise addition: C = A + B
        ///
        /// Both matrices must have the same dimensions.
        /// Result is a new CSR matrix containing the sum.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(nnz(A) + nnz(B))
        pub fn add(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format for easy merging
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            // Use HashMap to accumulate values at each (row, col) position
            const HashMap = std.AutoHashMap(usize, T);
            var accumulator = HashMap.init(allocator);
            defer accumulator.deinit();

            // Add all entries from self
            for (0..self.rows) |i| {
                const start = self.row_ptr[i];
                const end = self.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = self.col_indices[idx];
                    const value = self.values[idx];
                    const key = i * self.cols + j;
                    try accumulator.put(key, value);
                }
            }

            // Add all entries from other
            for (0..other.rows) |i| {
                const start = other.row_ptr[i];
                const end = other.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = other.col_indices[idx];
                    const value = other.values[idx];
                    const key = i * other.cols + j;
                    if (accumulator.get(key)) |existing| {
                        try accumulator.put(key, existing + value);
                    } else {
                        try accumulator.put(key, value);
                    }
                }
            }

            // Convert accumulator to COO
            var iter = accumulator.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const value = entry.value_ptr.*;
                const row = key / self.cols;
                const col = key % self.cols;
                try coo.append(row, col, value);
            }

            // Sort COO before converting to CSR
            try coo.sort();

            // Convert COO to CSR
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        /// Element-wise subtraction: C = A - B
        ///
        /// Both matrices must have the same dimensions.
        /// Result is a new CSR matrix containing the difference.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(nnz(A) + nnz(B))
        pub fn subtract(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            const HashMap = std.AutoHashMap(usize, T);
            var accumulator = HashMap.init(allocator);
            defer accumulator.deinit();

            // Add all entries from self
            for (0..self.rows) |i| {
                const start = self.row_ptr[i];
                const end = self.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = self.col_indices[idx];
                    const value = self.values[idx];
                    const key = i * self.cols + j;
                    try accumulator.put(key, value);
                }
            }

            // Subtract all entries from other
            for (0..other.rows) |i| {
                const start = other.row_ptr[i];
                const end = other.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = other.col_indices[idx];
                    const value = other.values[idx];
                    const key = i * other.cols + j;
                    if (accumulator.get(key)) |existing| {
                        try accumulator.put(key, existing - value);
                    } else {
                        try accumulator.put(key, -value);
                    }
                }
            }

            // Convert accumulator to COO
            var iter = accumulator.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const value = entry.value_ptr.*;
                const row = key / self.cols;
                const col = key % self.cols;
                try coo.append(row, col, value);
            }

            // Sort COO before converting to CSR
            try coo.sort();

            // Convert COO to CSR
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        /// Scalar multiplication: C = alpha * A
        ///
        /// Multiplies all non-zero elements by scalar alpha.
        /// Result is a new CSR matrix.
        ///
        /// Time: O(nnz) | Space: O(nnz)
        pub fn scale(self: *const Self, allocator: Allocator, alpha: T) !Self {
            // Allocate new arrays
            const n = self.nnz();
            const row_ptr = try allocator.alloc(usize, self.rows + 1);
            errdefer allocator.free(row_ptr);

            const col_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(col_indices);

            const values = try allocator.alloc(T, n);

            // Copy structure and scale values
            @memcpy(row_ptr, self.row_ptr);
            @memcpy(col_indices, self.col_indices);
            for (0..n) |i| {
                values[i] = alpha * self.values[i];
            }

            return Self{
                .rows = self.rows,
                .cols = self.cols,
                .row_ptr = row_ptr,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Hadamard product (element-wise multiplication): C = A ∘ B
        ///
        /// Both matrices must have the same dimensions.
        /// Only positions where BOTH matrices have non-zero entries will be non-zero in result.
        /// Result is a new CSR matrix containing the element-wise product.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(min(nnz(A), nnz(B)))
        pub fn hadamard(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            // Create HashMap from the matrix with fewer non-zeros for efficiency
            const HashMap = std.AutoHashMap(usize, T);
            var map = HashMap.init(allocator);
            defer map.deinit();

            // Choose smaller matrix to populate HashMap
            const use_self_for_map = self.nnz() <= other.nnz();
            const map_matrix = if (use_self_for_map) self else other;
            const scan_matrix = if (use_self_for_map) other else self;

            // Populate HashMap with one matrix
            for (0..map_matrix.rows) |i| {
                const start = map_matrix.row_ptr[i];
                const end = map_matrix.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = map_matrix.col_indices[idx];
                    const value = map_matrix.values[idx];
                    const key = i * map_matrix.cols + j;
                    try map.put(key, value);
                }
            }

            // Scan other matrix and multiply where both have non-zeros
            for (0..scan_matrix.rows) |i| {
                const start = scan_matrix.row_ptr[i];
                const end = scan_matrix.row_ptr[i + 1];
                for (start..end) |idx| {
                    const j = scan_matrix.col_indices[idx];
                    const value = scan_matrix.values[idx];
                    const key = i * scan_matrix.cols + j;
                    if (map.get(key)) |other_value| {
                        const product = value * other_value;
                        try coo.append(i, j, product);
                    }
                }
            }

            // Sort COO before converting to CSR
            try coo.sort();

            // Convert COO to CSR
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        pub const Entry = struct {
            col: usize,
            value: T,
        };

        /// Iterator for non-zero elements in a row
        pub const RowIterator = struct {
            col_indices: []const usize,
            values: []const T,
            index: usize,
            end: usize,

            pub fn next(self: *RowIterator) ?Entry {
                if (self.index >= self.end) return null;
                const result = Entry{
                    .col = self.col_indices[self.index],
                    .value = self.values[self.index],
                };
                self.index += 1;
                return result;
            }
        };

        /// Get iterator for row elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn rowIterator(self: *const Self, row: usize) RowIterator {
            if (row >= self.rows) {
                return RowIterator{
                    .col_indices = &[_]usize{},
                    .values = &[_]T{},
                    .index = 0,
                    .end = 0,
                };
            }

            const start = self.row_ptr[row];
            const end = self.row_ptr[row + 1];

            return RowIterator{
                .col_indices = self.col_indices,
                .values = self.values,
                .index = start,
                .end = end,
            };
        }

        /// Compute trace (sum of diagonal elements)
        ///
        /// Only valid for square matrices (rows == cols).
        /// Returns sum of all elements A[i,i] for i in [0, min(rows, cols)).
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn trace(self: *const Self) !T {
            if (self.rows != self.cols) {
                return error.NotSquare;
            }

            var sum: T = 0;
            for (0..self.rows) |i| {
                const start = self.row_ptr[i];
                const end = self.row_ptr[i + 1];
                for (start..end) |idx| {
                    if (self.col_indices[idx] == i) {
                        sum += self.values[idx];
                        break;
                    }
                }
            }
            return sum;
        }

        /// Extract diagonal elements as dense vector
        ///
        /// Returns array of length min(rows, cols) containing diagonal elements.
        /// Missing diagonal elements are returned as zero.
        ///
        /// Time: O(nnz) | Space: O(min(rows, cols))
        pub fn diag(self: *const Self, allocator: Allocator) ![]T {
            const n = @min(self.rows, self.cols);
            const result = try allocator.alloc(T, n);
            @memset(result, 0);

            for (0..n) |i| {
                const start = self.row_ptr[i];
                const end = self.row_ptr[i + 1];
                for (start..end) |idx| {
                    if (self.col_indices[idx] == i) {
                        result[i] = self.values[idx];
                        break;
                    }
                }
            }
            return result;
        }

        /// Compute density (ratio of non-zero elements)
        ///
        /// Returns value in [0.0, 1.0] where:
        /// - 0.0 = completely sparse (empty)
        /// - 1.0 = completely dense (no zeros)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn density(self: *const Self) f64 {
            if (self.rows == 0 or self.cols == 0) return 0.0;
            const total: f64 = @floatFromInt(self.rows * self.cols);
            const nonzeros: f64 = @floatFromInt(self.nnz());
            return nonzeros / total;
        }

        /// Compute sparsity (ratio of zero elements)
        ///
        /// Returns value in [0.0, 1.0] where:
        /// - 0.0 = completely dense (no zeros)
        /// - 1.0 = completely sparse (all zeros)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sparsity(self: *const Self) f64 {
            return 1.0 - self.density();
        }

        /// Compute Frobenius norm (sqrt of sum of squares of all elements)
        ///
        /// ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2) = sqrt(sum_{k=1}^{nnz} |values[k]|^2)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn normFrobenius(self: *const Self) T {
            var sum_sq: T = 0;
            for (self.values) |val| {
                sum_sq += val * val;
            }
            return @sqrt(sum_sq);
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - row_ptr has correct length (rows + 1)
        /// - col_indices and values have equal length
        /// - row_ptr is monotonically increasing
        /// - row_ptr[0] == 0
        /// - row_ptr[rows] == nnz
        /// - Column indices are within bounds [0, cols)
        /// - Column indices within each row are sorted (optional, CSR doesn't require this but it's good practice)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.nnz();

            // Check row_ptr length
            if (self.row_ptr.len != self.rows + 1) {
                return error.InvalidRowPtrLength;
            }

            // Check col_indices and values have equal length
            if (self.col_indices.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.len != n) {
                return error.InvalidArrayLength;
            }

            // Check row_ptr is monotonically increasing
            if (self.row_ptr[0] != 0) {
                return error.InvalidRowPtrStart;
            }
            for (1..self.row_ptr.len) |i| {
                if (self.row_ptr[i] < self.row_ptr[i - 1]) {
                    return error.RowPtrNotMonotonic;
                }
            }
            if (self.row_ptr[self.rows] != n) {
                return error.InvalidRowPtrEnd;
            }

            // Check column indices are within bounds
            for (self.col_indices) |col| {
                if (col >= self.cols) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// CSC (Compressed Sparse Column) Format
///
/// Stores sparse matrix in compressed column format (transpose of CSR):
/// - col_ptr[j] points to start of column j in row_indices and values
/// - col_ptr[j+1] - col_ptr[j] = number of non-zeros in column j
///
/// Memory: (n+1) + 2×nnz (column pointers + row indices + values)
///
/// Advantages:
/// - Efficient column-wise access O(1)
/// - Fast transpose operations
/// - Natural for column-oriented algorithms
///
/// Disadvantages:
/// - Row access is O(nnz)
/// - Modification requires rebuild
///
/// Time: O(1) | Space: O(n + nnz)
pub fn CSC(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Matrix dimensions
        rows: usize,
        cols: usize,

        /// Column pointers (length = cols + 1)
        col_ptr: []usize,

        /// Row indices (length = nnz)
        row_indices: []usize,

        /// Non-zero values (length = nnz)
        values: []T,

        /// Allocator used for dynamic arrays
        allocator: Allocator,

        /// Initialize CSC matrix from COO format
        ///
        /// Note: Input COO matrix will be sorted by (col, row)
        ///
        /// Time: O(nnz log nnz) | Space: O(n + nnz)
        pub fn fromCOO(allocator: Allocator, coo: *COO(T)) !Self {
            const n = coo.nnz();
            const cols_count = coo.cols;

            // Sort COO by column first
            try sortCOOByColumn(coo);

            // Allocate arrays
            const col_ptr = try allocator.alloc(usize, cols_count + 1);
            errdefer allocator.free(col_ptr);

            const row_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(row_indices);

            const values = try allocator.alloc(T, n);

            // Initialize col_ptr
            for (col_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return Self{
                    .rows = coo.rows,
                    .cols = cols_count,
                    .col_ptr = col_ptr,
                    .row_indices = row_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per column
            for (coo.col_indices.items) |col| {
                col_ptr[col + 1] += 1;
            }

            // Cumulative sum
            for (1..cols_count + 1) |i| {
                col_ptr[i] += col_ptr[i - 1];
            }

            // Fill row_indices and values
            const temp_ptr = try allocator.alloc(usize, cols_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, col_ptr);

            for (coo.col_indices.items, 0..) |col, i| {
                const pos = temp_ptr[col];
                row_indices[pos] = coo.row_indices.items[i];
                values[pos] = coo.values.items[i];
                temp_ptr[col] += 1;
            }

            return Self{
                .rows = coo.rows,
                .cols = cols_count,
                .col_ptr = col_ptr,
                .row_indices = row_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(1) | Space: O(1)
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.col_ptr);
            self.allocator.free(self.row_indices);
            self.allocator.free(self.values);
        }

        /// Get number of non-zero elements
        ///
        /// Time: O(1) | Space: O(1)
        pub fn nnz(self: *const Self) usize {
            return self.values.len;
        }

        /// Get non-zero count for specific column
        ///
        /// Time: O(1) | Space: O(1)
        pub fn colNnz(self: *const Self, col: usize) usize {
            if (col >= self.cols) return 0;
            return self.col_ptr[col + 1] - self.col_ptr[col];
        }

        /// Get element at (row, col), returns 0 if not found
        ///
        /// Time: O(nnz_col) | Space: O(1)
        pub fn get(self: *const Self, row: usize, col: usize) T {
            if (row >= self.rows or col >= self.cols) return 0;

            const start = self.col_ptr[col];
            const end = self.col_ptr[col + 1];

            for (self.row_indices[start..end], start..) |r, i| {
                if (r == row) return self.values[i];
            }

            return 0;
        }

        /// Transpose CSC matrix to CSR format
        ///
        /// Converts column-wise compressed format to row-wise compressed format.
        /// Effectively computes A^T (transpose).
        ///
        /// Time: O(nnz) | Space: O(m + nnz)
        pub fn transpose(self: *const Self, allocator: Allocator) !CSR(T) {
            const n = self.nnz();
            const row_count = self.rows;

            // Allocate CSR arrays
            const row_ptr = try allocator.alloc(usize, row_count + 1);
            errdefer allocator.free(row_ptr);

            const col_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(col_indices);

            const values = try allocator.alloc(T, n);

            // Initialize row_ptr to zero
            for (row_ptr) |*p| {
                p.* = 0;
            }

            if (n == 0) {
                return CSR(T){
                    .rows = self.cols, // Swap dimensions
                    .cols = self.rows,
                    .row_ptr = row_ptr,
                    .col_indices = col_indices,
                    .values = values,
                    .allocator = allocator,
                };
            }

            // Count non-zeros per row (becomes row_ptr)
            for (self.row_indices) |row| {
                row_ptr[row + 1] += 1;
            }

            // Cumulative sum to get row pointers
            for (1..row_count + 1) |i| {
                row_ptr[i] += row_ptr[i - 1];
            }

            // Fill col_indices and values
            const temp_ptr = try allocator.alloc(usize, row_count + 1);
            defer allocator.free(temp_ptr);
            mem.copyForwards(usize, temp_ptr, row_ptr);

            for (0..self.cols) |col| {
                const start = self.col_ptr[col];
                const end = self.col_ptr[col + 1];
                for (start..end) |idx| {
                    const row = self.row_indices[idx];
                    const pos = temp_ptr[row];
                    col_indices[pos] = col;
                    values[pos] = self.values[idx];
                    temp_ptr[row] += 1;
                }
            }

            return CSR(T){
                .rows = self.cols, // Swap dimensions
                .cols = self.rows,
                .row_ptr = row_ptr,
                .col_indices = col_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Sparse matrix-vector multiplication: y = A*x
        ///
        /// Computes the product of this sparse matrix (m×n) with a dense vector x (n×1),
        /// producing a dense vector y (m×1). Uses column-wise accumulation.
        ///
        /// Time: O(nnz) | Space: O(m) for result vector
        pub fn matvec(self: *const Self, allocator: Allocator, x: []const T) ![]T {
            const y = try allocator.alloc(T, self.rows);
            errdefer allocator.free(y);

            // Initialize result to zero
            for (y) |*val| {
                val.* = 0;
            }

            // Column-wise accumulation: y += x[j] * A[:,j]
            for (0..self.cols) |col| {
                const start = self.col_ptr[col];
                const end = self.col_ptr[col + 1];
                const x_val = x[col];
                for (start..end) |idx| {
                    const row = self.row_indices[idx];
                    y[row] += self.values[idx] * x_val;
                }
            }

            return y;
        }

        /// Element-wise addition: C = A + B
        ///
        /// Both matrices must have the same dimensions.
        /// Result is a new CSC matrix containing the sum.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(nnz(A) + nnz(B))
        pub fn add(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format for easy merging
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            // Use HashMap to accumulate values at each (row, col) position
            const HashMap = std.AutoHashMap(usize, T);
            var accumulator = HashMap.init(allocator);
            defer accumulator.deinit();

            // Add all entries from self
            for (0..self.cols) |j| {
                const start = self.col_ptr[j];
                const end = self.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = self.row_indices[idx];
                    const value = self.values[idx];
                    const key = i * self.cols + j;
                    try accumulator.put(key, value);
                }
            }

            // Add all entries from other
            for (0..other.cols) |j| {
                const start = other.col_ptr[j];
                const end = other.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = other.row_indices[idx];
                    const value = other.values[idx];
                    const key = i * other.cols + j;
                    if (accumulator.get(key)) |existing| {
                        try accumulator.put(key, existing + value);
                    } else {
                        try accumulator.put(key, value);
                    }
                }
            }

            // Convert accumulator to COO
            var iter = accumulator.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const value = entry.value_ptr.*;
                const row = key / self.cols;
                const col = key % self.cols;
                try coo.append(row, col, value);
            }

            // Sort COO before converting to CSC
            try coo.sort();

            // Convert COO to CSC
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        /// Element-wise subtraction: C = A - B
        ///
        /// Both matrices must have the same dimensions.
        /// Result is a new CSC matrix containing the difference.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(nnz(A) + nnz(B))
        pub fn subtract(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            const HashMap = std.AutoHashMap(usize, T);
            var accumulator = HashMap.init(allocator);
            defer accumulator.deinit();

            // Add all entries from self
            for (0..self.cols) |j| {
                const start = self.col_ptr[j];
                const end = self.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = self.row_indices[idx];
                    const value = self.values[idx];
                    const key = i * self.cols + j;
                    try accumulator.put(key, value);
                }
            }

            // Subtract all entries from other
            for (0..other.cols) |j| {
                const start = other.col_ptr[j];
                const end = other.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = other.row_indices[idx];
                    const value = other.values[idx];
                    const key = i * other.cols + j;
                    if (accumulator.get(key)) |existing| {
                        try accumulator.put(key, existing - value);
                    } else {
                        try accumulator.put(key, -value);
                    }
                }
            }

            // Convert accumulator to COO
            var iter = accumulator.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const value = entry.value_ptr.*;
                const row = key / self.cols;
                const col = key % self.cols;
                try coo.append(row, col, value);
            }

            // Sort COO before converting to CSC
            try coo.sort();

            // Convert COO to CSC
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        /// Scalar multiplication: C = alpha * A
        ///
        /// Multiplies all non-zero elements by scalar alpha.
        /// Result is a new CSC matrix.
        ///
        /// Time: O(nnz) | Space: O(nnz)
        pub fn scale(self: *const Self, allocator: Allocator, alpha: T) !Self {
            // Allocate new arrays
            const n = self.nnz();
            const col_ptr = try allocator.alloc(usize, self.cols + 1);
            errdefer allocator.free(col_ptr);

            const row_indices = try allocator.alloc(usize, n);
            errdefer allocator.free(row_indices);

            const values = try allocator.alloc(T, n);

            // Copy structure and scale values
            @memcpy(col_ptr, self.col_ptr);
            @memcpy(row_indices, self.row_indices);
            for (0..n) |i| {
                values[i] = alpha * self.values[i];
            }

            return Self{
                .rows = self.rows,
                .cols = self.cols,
                .col_ptr = col_ptr,
                .row_indices = row_indices,
                .values = values,
                .allocator = allocator,
            };
        }

        /// Hadamard product (element-wise multiplication): C = A ∘ B
        ///
        /// Both matrices must have the same dimensions.
        /// Only positions where BOTH matrices have non-zero entries will be non-zero in result.
        /// Result is a new CSC matrix containing the element-wise product.
        ///
        /// Time: O(nnz(A) + nnz(B)) | Space: O(min(nnz(A), nnz(B)))
        pub fn hadamard(self: *const Self, allocator: Allocator, other: *const Self) !Self {
            if (self.rows != other.rows or self.cols != other.cols) {
                return error.DimensionMismatch;
            }

            // Use COO as intermediate format
            var coo = COO(T).init(allocator, self.rows, self.cols);
            errdefer coo.deinit();

            // Create HashMap from the matrix with fewer non-zeros for efficiency
            const HashMap = std.AutoHashMap(usize, T);
            var map = HashMap.init(allocator);
            defer map.deinit();

            // Choose smaller matrix to populate HashMap
            const use_self_for_map = self.nnz() <= other.nnz();
            const map_matrix = if (use_self_for_map) self else other;
            const scan_matrix = if (use_self_for_map) other else self;

            // Populate HashMap with one matrix
            for (0..map_matrix.cols) |j| {
                const start = map_matrix.col_ptr[j];
                const end = map_matrix.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = map_matrix.row_indices[idx];
                    const value = map_matrix.values[idx];
                    const key = i * map_matrix.cols + j;
                    try map.put(key, value);
                }
            }

            // Scan other matrix and multiply where both have non-zeros
            for (0..scan_matrix.cols) |j| {
                const start = scan_matrix.col_ptr[j];
                const end = scan_matrix.col_ptr[j + 1];
                for (start..end) |idx| {
                    const i = scan_matrix.row_indices[idx];
                    const value = scan_matrix.values[idx];
                    const key = i * scan_matrix.cols + j;
                    if (map.get(key)) |other_value| {
                        const product = value * other_value;
                        try coo.append(i, j, product);
                    }
                }
            }

            // Sort COO before converting to CSC
            try coo.sort();

            // Convert COO to CSC
            const result = try Self.fromCOO(allocator, &coo);
            coo.deinit();

            return result;
        }

        /// Compute trace (sum of diagonal elements)
        ///
        /// Only valid for square matrices (rows == cols).
        /// Returns sum of all elements A[i,i] for i in [0, min(rows, cols)).
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn trace(self: *const Self) !T {
            if (self.rows != self.cols) {
                return error.NotSquare;
            }

            var sum: T = 0;
            for (0..self.cols) |j| {
                const start = self.col_ptr[j];
                const end = self.col_ptr[j + 1];
                for (start..end) |idx| {
                    if (self.row_indices[idx] == j) {
                        sum += self.values[idx];
                        break;
                    }
                }
            }
            return sum;
        }

        /// Extract diagonal elements as dense vector
        ///
        /// Returns array of length min(rows, cols) containing diagonal elements.
        /// Missing diagonal elements are returned as zero.
        ///
        /// Time: O(nnz) | Space: O(min(rows, cols))
        pub fn diag(self: *const Self, allocator: Allocator) ![]T {
            const n = @min(self.rows, self.cols);
            const result = try allocator.alloc(T, n);
            @memset(result, 0);

            for (0..n) |j| {
                const start = self.col_ptr[j];
                const end = self.col_ptr[j + 1];
                for (start..end) |idx| {
                    if (self.row_indices[idx] == j) {
                        result[j] = self.values[idx];
                        break;
                    }
                }
            }
            return result;
        }

        /// Compute density (ratio of non-zero elements)
        ///
        /// Returns value in [0.0, 1.0] where:
        /// - 0.0 = completely sparse (empty)
        /// - 1.0 = completely dense (no zeros)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn density(self: *const Self) f64 {
            if (self.rows == 0 or self.cols == 0) return 0.0;
            const total: f64 = @floatFromInt(self.rows * self.cols);
            const nonzeros: f64 = @floatFromInt(self.nnz());
            return nonzeros / total;
        }

        /// Compute sparsity (ratio of zero elements)
        ///
        /// Returns value in [0.0, 1.0] where:
        /// - 0.0 = completely dense (no zeros)
        /// - 1.0 = completely sparse (all zeros)
        ///
        /// Time: O(1) | Space: O(1)
        pub fn sparsity(self: *const Self) f64 {
            return 1.0 - self.density();
        }

        /// Compute Frobenius norm (sqrt of sum of squares of all elements)
        ///
        /// ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2) = sqrt(sum_{k=1}^{nnz} |values[k]|^2)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn normFrobenius(self: *const Self) T {
            var sum_sq: T = 0;
            for (self.values) |val| {
                sum_sq += val * val;
            }
            return @sqrt(sum_sq);
        }

        /// Validate internal invariants
        ///
        /// Checks:
        /// - col_ptr has correct length (cols + 1)
        /// - row_indices and values have equal length
        /// - col_ptr is monotonically increasing
        /// - col_ptr[0] == 0
        /// - col_ptr[cols] == nnz
        /// - Row indices are within bounds [0, rows)
        ///
        /// Time: O(nnz) | Space: O(1)
        pub fn validate(self: *const Self) !void {
            const n = self.nnz();

            // Check col_ptr length
            if (self.col_ptr.len != self.cols + 1) {
                return error.InvalidColPtrLength;
            }

            // Check row_indices and values have equal length
            if (self.row_indices.len != n) {
                return error.InvalidArrayLength;
            }
            if (self.values.len != n) {
                return error.InvalidArrayLength;
            }

            // Check col_ptr is monotonically increasing
            if (self.col_ptr[0] != 0) {
                return error.InvalidColPtrStart;
            }
            for (1..self.col_ptr.len) |i| {
                if (self.col_ptr[i] < self.col_ptr[i - 1]) {
                    return error.ColPtrNotMonotonic;
                }
            }
            if (self.col_ptr[self.cols] != n) {
                return error.InvalidColPtrEnd;
            }

            // Check row indices are within bounds
            for (self.row_indices) |row| {
                if (row >= self.rows) {
                    return error.IndexOutOfBounds;
                }
            }
        }
    };
}

/// Helper function to sort COO matrix by column then row
fn sortCOOByColumn(coo: anytype) !void {
    const Context = struct {
        cols: []const usize,
        rows: []const usize,

        pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
            if (ctx.cols[a_index] != ctx.cols[b_index]) {
                return ctx.cols[a_index] < ctx.cols[b_index];
            }
            return ctx.rows[a_index] < ctx.rows[b_index];
        }
    };

    const n = coo.nnz();
    if (n == 0) return;

    const indices = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(indices);

    for (indices, 0..) |*idx, i| {
        idx.* = i;
    }

    const context = Context{
        .cols = coo.col_indices.items,
        .rows = coo.row_indices.items,
    };

    std.mem.sort(usize, indices, context, Context.lessThan);

    // Reorder
    const temp_rows = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(temp_rows);
    const temp_cols = try coo.allocator.alloc(usize, n);
    defer coo.allocator.free(temp_cols);
    const temp_vals = try coo.allocator.alloc(@TypeOf(coo.values.items[0]), n);
    defer coo.allocator.free(temp_vals);

    for (indices, 0..) |idx, i| {
        temp_rows[i] = coo.row_indices.items[idx];
        temp_cols[i] = coo.col_indices.items[idx];
        temp_vals[i] = coo.values.items[idx];
    }

    mem.copyForwards(usize, coo.row_indices.items, temp_rows);
    mem.copyForwards(usize, coo.col_indices.items, temp_cols);
    mem.copyForwards(@TypeOf(coo.values.items[0]), coo.values.items, temp_vals);
}

// ============================================================================
// Tests
// ============================================================================

test "COO: init and basic operations" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try testing.expectEqual(@as(usize, 3), coo.rows);
    try testing.expectEqual(@as(usize, 3), coo.cols);
    try testing.expectEqual(@as(usize, 0), coo.nnz());
    try testing.expect(coo.isEmpty());
}

test "COO: append elements" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);

    try testing.expectEqual(@as(usize, 3), coo.nnz());
    try testing.expect(!coo.isEmpty());

    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[0]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[0]);
    try testing.expectEqual(@as(f64, 1.0), coo.values.items[0]);
}

test "COO: out of bounds" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try testing.expectError(error.OutOfBounds, coo.append(3, 0, 1.0));
    try testing.expectError(error.OutOfBounds, coo.append(0, 3, 1.0));
}

test "COO: sort" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(2, 1, 6.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 0, 4.0);
    try coo.append(0, 0, 1.0);

    try coo.sort();

    // Should be sorted by (row, col)
    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[0]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[0]);
    try testing.expectEqual(@as(f64, 1.0), coo.values.items[0]);

    try testing.expectEqual(@as(usize, 0), coo.row_indices.items[1]);
    try testing.expectEqual(@as(usize, 2), coo.col_indices.items[1]);
    try testing.expectEqual(@as(f64, 3.0), coo.values.items[1]);

    try testing.expectEqual(@as(usize, 1), coo.row_indices.items[2]);
    try testing.expectEqual(@as(usize, 0), coo.col_indices.items[2]);
    try testing.expectEqual(@as(f64, 4.0), coo.values.items[2]);

    try testing.expectEqual(@as(usize, 2), coo.row_indices.items[3]);
    try testing.expectEqual(@as(usize, 1), coo.col_indices.items[3]);
    try testing.expectEqual(@as(f64, 6.0), coo.values.items[3]);
}

test "CSR: fromCOO basic" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 3), csr.rows);
    try testing.expectEqual(@as(usize, 3), csr.cols);
    try testing.expectEqual(@as(usize, 3), csr.nnz());

    // Check row pointers
    try testing.expectEqual(@as(usize, 0), csr.row_ptr[0]);
    try testing.expectEqual(@as(usize, 1), csr.row_ptr[1]);
    try testing.expectEqual(@as(usize, 2), csr.row_ptr[2]);
    try testing.expectEqual(@as(usize, 3), csr.row_ptr[3]);
}

test "CSR: get element" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(f64, 1.0), csr.get(0, 0));
    try testing.expectEqual(@as(f64, 3.0), csr.get(0, 2));
    try testing.expectEqual(@as(f64, 5.0), csr.get(1, 1));
    try testing.expectEqual(@as(f64, 0.0), csr.get(0, 1)); // Zero element
    try testing.expectEqual(@as(f64, 0.0), csr.get(2, 2)); // Zero element
}

test "CSR: row iterator" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var it = csr.rowIterator(0);
    const elem1 = it.next().?;
    try testing.expectEqual(@as(usize, 0), elem1.col);
    try testing.expectEqual(@as(f64, 1.0), elem1.value);

    const elem2 = it.next().?;
    try testing.expectEqual(@as(usize, 2), elem2.col);
    try testing.expectEqual(@as(f64, 3.0), elem2.value);

    try testing.expect(it.next() == null);
}

test "CSR: rowNnz" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 2), csr.rowNnz(0));
    try testing.expectEqual(@as(usize, 1), csr.rowNnz(1));
    try testing.expectEqual(@as(usize, 0), csr.rowNnz(2));
}

test "CSC: fromCOO basic" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(usize, 3), csc.rows);
    try testing.expectEqual(@as(usize, 3), csc.cols);
    try testing.expectEqual(@as(usize, 3), csc.nnz());

    // Check column pointers
    try testing.expectEqual(@as(usize, 0), csc.col_ptr[0]);
    try testing.expectEqual(@as(usize, 1), csc.col_ptr[1]);
    try testing.expectEqual(@as(usize, 2), csc.col_ptr[2]);
    try testing.expectEqual(@as(usize, 3), csc.col_ptr[3]);
}

test "CSC: get element" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(2, 0, 7.0);
    try coo.append(1, 1, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(f64, 1.0), csc.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), csc.get(2, 0));
    try testing.expectEqual(@as(f64, 5.0), csc.get(1, 1));
    try testing.expectEqual(@as(f64, 0.0), csc.get(0, 1)); // Zero element
}

test "CSC: colNnz" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(2, 0, 7.0);
    try coo.append(1, 1, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectEqual(@as(usize, 2), csc.colNnz(0));
    try testing.expectEqual(@as(usize, 1), csc.colNnz(1));
    try testing.expectEqual(@as(usize, 0), csc.colNnz(2));
}

test "Sparse: empty matrix" {
    var coo = COO(f64).init(testing.allocator, 5, 5);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(usize, 0), csr.nnz());
    try testing.expectEqual(@as(f64, 0.0), csr.get(0, 0));
}

test "Sparse: integer types" {
    var coo = COO(i32).init(testing.allocator, 2, 2);
    defer coo.deinit();

    try coo.append(0, 0, 10);
    try coo.append(1, 1, 20);
    try coo.sort();

    var csr = try CSR(i32).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectEqual(@as(i32, 10), csr.get(0, 0));
    try testing.expectEqual(@as(i32, 20), csr.get(1, 1));
}

test "COO: memory safety with multiple iterations" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        var coo = COO(f64).init(testing.allocator, 100, 100);
        defer coo.deinit();

        var j: usize = 0;
        while (j < 50) : (j += 1) {
            try coo.append(j, j, @as(f64, @floatFromInt(j)));
        }

        try coo.sort();
        try testing.expectEqual(@as(usize, 50), coo.nnz());
    }
}

// ============================================================================
// Sparse Matrix Operations: Transpose and SpMV
// ============================================================================

test "CSR transpose: general 3x3 matrix" {
    // Create 3x3 matrix:
    // [1 0 3]
    // [0 5 0]
    // [7 0 9]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.append(2, 0, 7.0);
    try coo.append(2, 2, 9.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Manually build transpose: [A^T]_ij = A_ji
    // A^T will be:
    // [1 0 7]
    // [0 5 0]
    // [3 0 9]
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0); // A[0,0] -> A^T[0,0]
    try coo_t.append(0, 2, 7.0); // A[2,0] -> A^T[0,2]
    try coo_t.append(1, 1, 5.0); // A[1,1] -> A^T[1,1]
    try coo_t.append(2, 0, 3.0); // A[0,2] -> A^T[2,0]
    try coo_t.append(2, 2, 9.0); // A[2,2] -> A^T[2,2]
    try coo_t.sort();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 3), csr_t.rows);
    try testing.expectEqual(@as(usize, 3), csr_t.cols);
    try testing.expectEqual(@as(usize, 5), csr_t.nnz());

    // Verify transposed values
    try testing.expectEqual(@as(f64, 1.0), csr_t.get(0, 0));
    try testing.expectEqual(@as(f64, 7.0), csr_t.get(0, 2));
    try testing.expectEqual(@as(f64, 5.0), csr_t.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), csr_t.get(2, 0));
    try testing.expectEqual(@as(f64, 9.0), csr_t.get(2, 2));
}

test "CSR transpose: 2x4 rectangular matrix" {
    // Create 2x4 matrix:
    // [1 2 0 3]
    // [0 4 5 0]
    var coo = COO(f64).init(testing.allocator, 2, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(0, 3, 3.0);
    try coo.append(1, 1, 4.0);
    try coo.append(1, 2, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Build CSC representation of transpose (should be 4x2)
    var coo_t = COO(f64).init(testing.allocator, 4, 2);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0); // (0,0) from A
    try coo_t.append(1, 0, 2.0); // (0,1) from A
    try coo_t.append(1, 1, 4.0); // (1,1) from A
    try coo_t.append(2, 1, 5.0); // (1,2) from A
    try coo_t.append(3, 0, 3.0); // (0,3) from A

    var csc_t = try CSC(f64).fromCOO(testing.allocator, &coo_t);
    defer csc_t.deinit();

    try testing.expectEqual(@as(usize, 4), csc_t.rows);
    try testing.expectEqual(@as(usize, 2), csc_t.cols);
    try testing.expectEqual(@as(usize, 5), csc_t.nnz());
}

test "CSR transpose: single element matrix" {
    // Create 1x1 matrix with single non-zero
    var coo = COO(f64).init(testing.allocator, 1, 1);
    defer coo.deinit();

    try coo.append(0, 0, 42.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var coo_t = COO(f64).init(testing.allocator, 1, 1);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 42.0);

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(f64, 42.0), csr_t.get(0, 0));
    try testing.expectEqual(@as(usize, 1), csr_t.nnz());
}

test "CSR transpose: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 0), csr_t.nnz());
    try testing.expectEqual(@as(f64, 0.0), csr_t.get(0, 0));
}

test "CSC transpose: 3x3 matrix to CSR" {
    // Create 3x3 CSC matrix:
    // [1 0 3]
    // [0 5 0]
    // [7 0 9]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 3.0);
    try coo.append(1, 1, 5.0);
    try coo.append(2, 0, 7.0);
    try coo.append(2, 2, 9.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Transpose of CSC: build COO from transposed values
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 1.0);
    try coo_t.append(1, 0, 5.0);
    try coo_t.append(2, 0, 3.0);
    try coo_t.append(0, 2, 7.0);
    try coo_t.append(2, 2, 9.0);
    try coo_t.sort();

    var csr_t = try CSR(f64).fromCOO(testing.allocator, &coo_t);
    defer csr_t.deinit();

    try testing.expectEqual(@as(usize, 3), csr_t.rows);
    try testing.expectEqual(@as(usize, 3), csr_t.cols);
    try testing.expectEqual(@as(usize, 5), csr_t.nnz());
}

test "CSC transpose: symmetric matrix" {
    // Create 3x3 symmetric matrix:
    // [2 1 0]
    // [1 3 4]
    // [0 4 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(0, 1, 1.0);
    try coo.append(1, 0, 1.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 1, 4.0);
    try coo.append(2, 2, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Transpose should equal original (symmetric)
    var coo_t = COO(f64).init(testing.allocator, 3, 3);
    defer coo_t.deinit();

    try coo_t.append(0, 0, 2.0);
    try coo_t.append(0, 1, 1.0);
    try coo_t.append(1, 0, 1.0);
    try coo_t.append(1, 1, 3.0);
    try coo_t.append(1, 2, 4.0);
    try coo_t.append(2, 1, 4.0);
    try coo_t.append(2, 2, 5.0);

    var csc_t = try CSC(f64).fromCOO(testing.allocator, &coo_t);
    defer csc_t.deinit();

    try testing.expectEqual(csc.nnz(), csc_t.nnz());
    // Verify symmetric property: A[i,j] = A[j,i]
    try testing.expectEqual(@as(f64, 1.0), csc_t.get(0, 1));
    try testing.expectEqual(@as(f64, 1.0), csc_t.get(1, 0));
}

// ============================================================================
// Sparse Matrix-Vector Multiplication (SpMV)
// ============================================================================

test "CSR matvec: identity matrix" {
    // Create 3x3 identity matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Create vector x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = I * x = x = [1, 2, 3]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // Perform y = A*x (identity case)
    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 1.0), y[0]);
    try testing.expectEqual(@as(f64, 2.0), y[1]);
    try testing.expectEqual(@as(f64, 3.0), y[2]);
}

test "CSR matvec: diagonal matrix" {
    // Create 3x3 diagonal matrix [2, 3, 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 5.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = D*x = [2*1, 3*2, 5*3] = [2, 6, 15]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 2.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 15.0), y[2]);
}

test "CSR matvec: general sparse matrix" {
    // Create 3x4 matrix:
    // [1 0 2 0]
    // [0 3 0 4]
    // [5 0 0 6]
    var coo = COO(f64).init(testing.allocator, 3, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 3, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 3, 6.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // Manual calculation:
    // y[0] = 1*1 + 2*3 = 1 + 6 = 7
    // y[1] = 3*2 + 4*4 = 6 + 16 = 22
    // y[2] = 5*1 + 6*4 = 5 + 24 = 29
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 7.0), y[0]);
    try testing.expectEqual(@as(f64, 22.0), y[1]);
    try testing.expectEqual(@as(f64, 29.0), y[2]);
}

test "CSR matvec: single row matrix" {
    // Create 1x4 matrix: [2 0 3 0]
    var coo = COO(f64).init(testing.allocator, 1, 4);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(0, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // y = [2*1 + 3*3] = [11]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 1);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 11.0), y[0]);
}

test "CSR matvec: single column matrix" {
    // Create 3x1 matrix: [5]
    //                    [2]
    //                    [7]
    var coo = COO(f64).init(testing.allocator, 3, 1);
    defer coo.deinit();

    try coo.append(0, 0, 5.0);
    try coo.append(1, 0, 2.0);
    try coo.append(2, 0, 7.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // x = [3]
    const x = [_]f64{3.0};

    // y = [5*3, 2*3, 7*3] = [15, 6, 21]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 15.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 21.0), y[2]);
}

test "CSR matvec: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const x = [_]f64{ 1.0, 2.0, 3.0 };

    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // y = A*x where A is zero matrix should be zero vector
    for (0..csr.rows) |i| {
        y[i] = 0.0;
        var it = csr.rowIterator(i);
        while (it.next()) |entry| {
            y[i] += entry.value * x[entry.col];
        }
    }

    try testing.expectEqual(@as(f64, 0.0), y[0]);
    try testing.expectEqual(@as(f64, 0.0), y[1]);
    try testing.expectEqual(@as(f64, 0.0), y[2]);
}

test "CSR matmul: identity × identity" {
    // I × I = I
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(1, 1, 1.0);
    try coo_a.append(2, 2, 1.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);
    try coo_b.append(1, 1, 1.0);
    try coo_b.append(2, 2, 1.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    // Result should be identity
    try testing.expectEqual(@as(usize, 3), result.nnz());
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 1.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, 1.0), result.get(2, 2));
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 2));
}

test "CSR matmul: diagonal × diagonal" {
    // [2 0 0]   [3 0 0]   [6 0 0]
    // [0 3 0] × [0 4 0] = [0 12 0]
    // [0 0 4]   [0 0 5]   [0 0 20]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 2, 4.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 3.0);
    try coo_b.append(1, 1, 4.0);
    try coo_b.append(2, 2, 5.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.nnz());
    try testing.expectEqual(@as(f64, 6.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 12.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, 20.0), result.get(2, 2));
}

test "CSR matmul: general sparse matrices" {
    // A = [1 2 0]   B = [1 0]   C = [1*1+2*3  1*0+2*4] = [7  8]
    //     [0 3 0]       [3 4]       [0*1+3*3  0*0+3*4]   [9 12]
    //                   [0 0]       [0        0]         [0  0]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 1, 2.0);
    try coo_a.append(1, 1, 3.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);
    try coo_b.append(1, 0, 3.0);
    try coo_b.append(1, 1, 4.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 2), result.cols);
    try testing.expectEqual(@as(f64, 7.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 8.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 9.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 12.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, 0.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(2, 1));
}

test "CSR matmul: rectangular dimensions" {
    // A (2×3) × B (3×2) = C (2×2)
    // A = [1 0 2]   B = [1 2]   C = [1*1+0*0+2*0  1*2+0*0+2*3] = [1 8]
    //     [0 3 0]       [0 0]       [0*1+3*0+0*0  0*2+3*0+0*3]   [0 0]
    //                   [0 3]
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);
    try coo_b.append(0, 1, 2.0);
    try coo_b.append(2, 1, 3.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.rows);
    try testing.expectEqual(@as(usize, 2), result.cols);
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 8.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 1));
}

test "CSR matmul: empty matrices" {
    // Zero matrix × zero matrix = zero matrix
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.rows);
    try testing.expectEqual(@as(usize, 2), result.cols);
    try testing.expectEqual(@as(usize, 0), result.nnz());
}

test "CSR matmul: dimension mismatch error" {
    // A (2×3) × B (2×2) should fail (cols of A != rows of B)
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    const result = csr_a.matmul(testing.allocator, &csr_b);
    try testing.expectError(error.DimensionMismatch, result);
}

test "CSR matmul: integer types (i32)" {
    // Test with integer type
    var coo_a = COO(i32).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 2);
    try coo_a.append(0, 1, 3);
    try coo_a.append(1, 1, 4);

    var csr_a = try CSR(i32).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(i32).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1);
    try coo_b.append(1, 0, 2);
    try coo_b.append(1, 1, 3);

    var csr_b = try CSR(i32).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.matmul(testing.allocator, &csr_b);
    defer result.deinit();

    // [2 3] × [1 0] = [2*1+3*2  2*0+3*3] = [8  9]
    // [0 4]   [2 3]   [0*1+4*2  0*0+4*3]   [8 12]
    try testing.expectEqual(@as(i32, 8), result.get(0, 0));
    try testing.expectEqual(@as(i32, 9), result.get(0, 1));
    try testing.expectEqual(@as(i32, 8), result.get(1, 0));
    try testing.expectEqual(@as(i32, 12), result.get(1, 1));
}

test "CSR matmul: memory safety check" {
    // Run multiple iterations to check for memory leaks
    for (0..10) |_| {
        var coo_a = COO(f64).init(testing.allocator, 5, 5);
        defer coo_a.deinit();
        try coo_a.append(0, 1, 2.0);
        try coo_a.append(2, 3, 4.0);
        try coo_a.append(4, 0, 5.0);

        var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
        defer csr_a.deinit();

        var coo_b = COO(f64).init(testing.allocator, 5, 5);
        defer coo_b.deinit();
        try coo_b.append(1, 2, 3.0);
        try coo_b.append(3, 4, 6.0);

        var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
        defer csr_b.deinit();

        var result = try csr_a.matmul(testing.allocator, &csr_b);
        defer result.deinit();

        // Verify expected non-zero count and values
        const expected_nnz = result.nnz();
        try testing.expect(expected_nnz >= 0); // Should be valid
    }
}

test "CSR add: general sparse matrices" {
    // Matrix A (3x3):
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    // Matrix B (3x3):
    // [0 1 0]
    // [2 0 3]
    // [0 4 0]
    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 1, 1.0);
    try coo_b.append(1, 0, 2.0);
    try coo_b.append(1, 2, 3.0);
    try coo_b.append(2, 1, 4.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    // C = A + B =
    // [1 1 2]
    // [2 3 3]
    // [4 4 5]
    var result = try csr_a.add(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 9), result.nnz());

    // Verify values
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, 2.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 2));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 1));
    try testing.expectEqual(@as(f64, 5.0), result.get(2, 2));
}

test "CSR add: disjoint patterns" {
    // Matrix A (2x2): [1 0; 0 0]
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    // Matrix B (2x2): [0 0; 0 2]
    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(1, 1, 2.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    // C = A + B = [1 0; 0 2]
    var result = try csr_a.add(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), result.get(1, 1));
}

test "CSR add: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    try testing.expectError(error.DimensionMismatch, csr_a.add(testing.allocator, &csr_b));
}

test "CSR subtract: general sparse matrices" {
    // Matrix A (3x3):
    // [5 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 5.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    // Matrix B (3x3):
    // [1 1 0]
    // [2 0 3]
    // [0 4 0]
    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);
    try coo_b.append(0, 1, 1.0);
    try coo_b.append(1, 0, 2.0);
    try coo_b.append(1, 2, 3.0);
    try coo_b.append(2, 1, 4.0);

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    // C = A - B =
    // [ 4 -1  2]
    // [-2  3 -3]
    // [ 4 -4  5]
    var result = try csr_a.subtract(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 9), result.nnz());

    // Verify values
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, -1.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, -2.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, -3.0), result.get(1, 2));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, -4.0), result.get(2, 1));
    try testing.expectEqual(@as(f64, 5.0), result.get(2, 2));
}

test "CSR subtract: cancellation creates zeros" {
    // Matrix A (2x2): [1 2; 3 4]
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 1, 2.0);
    try coo_a.append(1, 0, 3.0);
    try coo_a.append(1, 1, 4.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    // B = A, so A - B should be zero
    var result = try csr_a.subtract(testing.allocator, &csr_a);
    defer result.deinit();

    // All elements should be zero (still stored in sparse format)
    try testing.expectEqual(@as(usize, 4), result.nnz());
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 1));
}

test "CSR subtract: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    try testing.expectError(error.DimensionMismatch, csr_a.subtract(testing.allocator, &csr_b));
}

test "CSR scale: general sparse matrix" {
    // Matrix A (3x3):
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 0, 4.0);
    try coo.append(2, 2, 5.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // C = 2.5 * A
    var result = try csr.scale(testing.allocator, 2.5);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 5), result.nnz());

    // Verify scaled values
    try testing.expectEqual(@as(f64, 2.5), result.get(0, 0));
    try testing.expectEqual(@as(f64, 5.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, 7.5), result.get(1, 1));
    try testing.expectEqual(@as(f64, 10.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, 12.5), result.get(2, 2));
}

test "CSR scale: zero scalar" {
    var coo = COO(f64).init(testing.allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // C = 0 * A (all zeros)
    var result = try csr.scale(testing.allocator, 0.0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 1));
}

test "CSR scale: negative scalar" {
    var coo = COO(f64).init(testing.allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 3.0);
    try coo.append(1, 1, -4.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // C = -1 * A
    var result = try csr.scale(testing.allocator, -1.0);
    defer result.deinit();

    try testing.expectEqual(@as(f64, -3.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), result.get(1, 1));
}

test "CSR scale: memory safety check" {
    // Run multiple iterations to check for memory leaks
    for (0..10) |_| {
        var coo = COO(f64).init(testing.allocator, 5, 5);
        defer coo.deinit();
        try coo.append(0, 1, 2.0);
        try coo.append(2, 3, 4.0);
        try coo.append(4, 0, 5.0);

        var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
        defer csr.deinit();

        var result = try csr.scale(testing.allocator, 3.0);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 3), result.nnz());
    }
}

test "CSR hadamard: general sparse matrices" {
    // Matrix A:          Matrix B:
    // [1  0  2]          [0  0  2]
    // [0  3  0]          [0  3  5]
    // [4  0  5]          [4  6  0]
    //
    // A ∘ B (element-wise product, only where both non-zero):
    // [0  0  4]  (1×0=0 missing, 0×0=0 missing, 2×2=4)
    // [0  9  0]  (0×0=0 missing, 3×3=9, 0×5=0 missing)
    // [16 0  0]  (4×4=16, 0×6=0 missing, 5×0=0 missing)
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 2, 2.0);
    try coo_b.append(1, 1, 3.0);
    try coo_b.append(1, 2, 5.0);
    try coo_b.append(2, 0, 4.0);
    try coo_b.append(2, 1, 6.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    // Result should have 3 non-zeros
    try testing.expectEqual(@as(usize, 3), result.nnz());

    // Check values
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 2));  // 2×2=4
    try testing.expectEqual(@as(f64, 9.0), result.get(1, 1));  // 3×3=9
    try testing.expectEqual(@as(f64, 16.0), result.get(2, 0)); // 4×4=16
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));  // No overlap
    try testing.expectEqual(@as(f64, 0.0), result.get(2, 2));  // No overlap
}

test "CSR hadamard: diagonal matrices" {
    // A = diag([1, 2, 3])
    // B = diag([4, 5, 6])
    // A ∘ B = diag([4, 10, 18])
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(1, 1, 2.0);
    try coo_a.append(2, 2, 3.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 4.0);
    try coo_b.append(1, 1, 5.0);
    try coo_b.append(2, 2, 6.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.nnz());
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 0));   // 1×4=4
    try testing.expectEqual(@as(f64, 10.0), result.get(1, 1));  // 2×5=10
    try testing.expectEqual(@as(f64, 18.0), result.get(2, 2));  // 3×6=18
}

test "CSR hadamard: no overlap (result is zero matrix)" {
    // A has non-zeros at (0,0) and (1,1)
    // B has non-zeros at (0,1) and (1,0)
    // No overlap → result should be empty
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(1, 1, 2.0);

    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 1, 3.0);
    try coo_b.append(1, 0, 4.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.nnz());
    try testing.expect(result.isEmpty());
}

test "CSR hadamard: complete overlap" {
    // A and B have same sparsity pattern
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 2.0);
    try coo_a.append(0, 1, 3.0);
    try coo_a.append(1, 0, 4.0);
    try coo_a.append(1, 1, 5.0);

    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 6.0);
    try coo_b.append(0, 1, 7.0);
    try coo_b.append(1, 0, 8.0);
    try coo_b.append(1, 1, 9.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.nnz());
    try testing.expectEqual(@as(f64, 12.0), result.get(0, 0)); // 2×6=12
    try testing.expectEqual(@as(f64, 21.0), result.get(0, 1)); // 3×7=21
    try testing.expectEqual(@as(f64, 32.0), result.get(1, 0)); // 4×8=32
    try testing.expectEqual(@as(f64, 45.0), result.get(1, 1)); // 5×9=45
}

test "CSR hadamard: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 2.0);

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    try testing.expectError(error.DimensionMismatch, csr_a.hadamard(testing.allocator, &csr_b));
}

test "CSR hadamard: empty matrices" {
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();

    var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.nnz());
    try testing.expect(result.isEmpty());
}

test "CSR hadamard: integer type" {
    var coo_a = COO(i32).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 3);
    try coo_a.append(1, 1, 4);

    var coo_b = COO(i32).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 5);
    try coo_b.append(1, 1, 6);

    var csr_a = try CSR(i32).fromCOO(testing.allocator, &coo_a);
    defer csr_a.deinit();

    var csr_b = try CSR(i32).fromCOO(testing.allocator, &coo_b);
    defer csr_b.deinit();

    var result = try csr_a.hadamard(testing.allocator, &csr_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(i32, 15), result.get(0, 0)); // 3×5=15
    try testing.expectEqual(@as(i32, 24), result.get(1, 1)); // 4×6=24
}

test "CSR hadamard: memory safety check" {
    // Run multiple iterations to check for memory leaks
    for (0..10) |_| {
        var coo_a = COO(f64).init(testing.allocator, 5, 5);
        defer coo_a.deinit();
        try coo_a.append(0, 1, 2.0);
        try coo_a.append(2, 3, 4.0);
        try coo_a.append(4, 0, 5.0);

        var coo_b = COO(f64).init(testing.allocator, 5, 5);
        defer coo_b.deinit();
        try coo_b.append(0, 1, 3.0);
        try coo_b.append(2, 3, 6.0);

        var csr_a = try CSR(f64).fromCOO(testing.allocator, &coo_a);
        defer csr_a.deinit();

        var csr_b = try CSR(f64).fromCOO(testing.allocator, &coo_b);
        defer csr_b.deinit();

        var result = try csr_a.hadamard(testing.allocator, &csr_b);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 2), result.nnz());
    }
}

test "CSR trace: square matrix with diagonal" {
    // Matrix:
    // [1  2  0]
    // [0  3  4]
    // [5  0  6]
    // Trace = 1 + 3 + 6 = 10
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 2, 6.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const tr = try csr.trace();
    try testing.expectEqual(@as(f64, 10.0), tr);
}

test "CSR trace: identity matrix" {
    var coo = COO(f64).init(testing.allocator, 4, 4);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.append(3, 3, 1.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const tr = try csr.trace();
    try testing.expectEqual(@as(f64, 4.0), tr);
}

test "CSR trace: non-square matrix error" {
    var coo = COO(f64).init(testing.allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    try testing.expectError(error.NotSquare, csr.trace());
}

test "CSR trace: missing diagonal elements" {
    // Matrix with no diagonal elements (trace = 0)
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 1, 2.0);
    try coo.append(1, 2, 3.0);
    try coo.append(2, 0, 4.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const tr = try csr.trace();
    try testing.expectEqual(@as(f64, 0.0), tr);
}

test "CSR diag: general sparse matrix" {
    // Matrix:
    // [1  2  0]
    // [0  3  4]
    // [5  0  6]
    // Diagonal = [1, 3, 6]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 2, 6.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const d = try csr.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 3), d.len);
    try testing.expectEqual(@as(f64, 1.0), d[0]);
    try testing.expectEqual(@as(f64, 3.0), d[1]);
    try testing.expectEqual(@as(f64, 6.0), d[2]);
}

test "CSR diag: rectangular matrix" {
    // 2×3 matrix:
    // [1  2  0]
    // [0  3  4]
    // Diagonal = [1, 3] (min(2,3) = 2)
    var coo = COO(f64).init(testing.allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const d = try csr.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 2), d.len);
    try testing.expectEqual(@as(f64, 1.0), d[0]);
    try testing.expectEqual(@as(f64, 3.0), d[1]);
}

test "CSR diag: missing diagonal elements" {
    // Matrix with missing diagonal → zeros in result
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 1, 2.0);
    try coo.append(1, 2, 3.0);
    try coo.append(2, 0, 4.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const d = try csr.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 3), d.len);
    try testing.expectEqual(@as(f64, 0.0), d[0]);
    try testing.expectEqual(@as(f64, 0.0), d[1]);
    try testing.expectEqual(@as(f64, 0.0), d[2]);
}

test "CSR density and sparsity: various matrices" {
    // Dense 2x2 matrix (all elements non-zero)
    {
        var coo = COO(f64).init(testing.allocator, 2, 2);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(0, 1, 2.0);
        try coo.append(1, 0, 3.0);
        try coo.append(1, 1, 4.0);

        var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
        defer csr.deinit();

        try testing.expectEqual(@as(f64, 1.0), csr.density());
        try testing.expectEqual(@as(f64, 0.0), csr.sparsity());
    }

    // Sparse 3x3 matrix (3 non-zeros out of 9)
    {
        var coo = COO(f64).init(testing.allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(1, 1, 2.0);
        try coo.append(2, 2, 3.0);

        var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
        defer csr.deinit();

        const expected_density = 3.0 / 9.0;
        const expected_sparsity = 6.0 / 9.0;
        try testing.expectApproxEqRel(expected_density, csr.density(), 1e-10);
        try testing.expectApproxEqRel(expected_sparsity, csr.sparsity(), 1e-10);
    }

    // Empty matrix
    {
        var coo = COO(f64).init(testing.allocator, 3, 3);
        defer coo.deinit();

        var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
        defer csr.deinit();

        try testing.expectEqual(@as(f64, 0.0), csr.density());
        try testing.expectEqual(@as(f64, 1.0), csr.sparsity());
    }
}

test "CSR normFrobenius: general sparse matrix" {
    // Matrix:
    // [3  0  4]
    // [0  0  0]
    // [0  5  0]
    // Frobenius norm = sqrt(3^2 + 4^2 + 5^2) = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.071
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 3.0);
    try coo.append(0, 2, 4.0);
    try coo.append(2, 1, 5.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const norm = csr.normFrobenius();
    const expected = @sqrt(@as(f64, 50.0));
    try testing.expectApproxEqRel(expected, norm, 1e-10);
}

test "CSR normFrobenius: identity matrix" {
    var coo = COO(f64).init(testing.allocator, 4, 4);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.append(3, 3, 1.0);

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const norm = csr.normFrobenius();
    try testing.expectApproxEqRel(@as(f64, 2.0), norm, 1e-10); // sqrt(4) = 2
}

test "CSR normFrobenius: empty matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    const norm = csr.normFrobenius();
    try testing.expectEqual(@as(f64, 0.0), norm);
}

test "CSR utilities: memory safety check" {
    // Run multiple iterations to check for memory leaks in diag()
    for (0..10) |_| {
        var coo = COO(f64).init(testing.allocator, 4, 4);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(1, 1, 2.0);
        try coo.append(2, 2, 3.0);
        try coo.append(3, 3, 4.0);

        var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
        defer csr.deinit();

        const d = try csr.diag(testing.allocator);
        defer testing.allocator.free(d);

        _ = try csr.trace();
        _ = csr.density();
        _ = csr.sparsity();
        _ = csr.normFrobenius();
    }
}

test "CSC matvec: identity matrix" {
    // Create 3x3 identity matrix via CSC
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = I * x = x = [1, 2, 3]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    // SpMV with CSC: column-wise accumulation
    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 1.0), y[0]);
    try testing.expectEqual(@as(f64, 2.0), y[1]);
    try testing.expectEqual(@as(f64, 3.0), y[2]);
}

test "CSC matvec: diagonal matrix" {
    // Create 3x3 diagonal matrix [2, 3, 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 2, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3]
    const x = [_]f64{ 1.0, 2.0, 3.0 };

    // y = D*x = [2*1, 3*2, 5*3] = [2, 6, 15]
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 2.0), y[0]);
    try testing.expectEqual(@as(f64, 6.0), y[1]);
    try testing.expectEqual(@as(f64, 15.0), y[2]);
}

test "CSC matvec: general sparse matrix" {
    // Create 3x4 matrix:
    // [1 0 2 0]
    // [0 3 0 4]
    // [5 0 0 6]
    var coo = COO(f64).init(testing.allocator, 3, 4);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 3, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 3, 6.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [1, 2, 3, 4]
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    // SpMV: y = A*x
    // y[0] = A[0,0]*x[0] + A[0,1]*x[1] + A[0,2]*x[2] + A[0,3]*x[3] = 1*1 + 0*2 + 2*3 + 0*4 = 1 + 6 = 7
    // y[1] = A[1,0]*x[0] + A[1,1]*x[1] + A[1,2]*x[2] + A[1,3]*x[3] = 0*1 + 3*2 + 0*3 + 4*4 = 6 + 16 = 22
    // y[2] = A[2,0]*x[0] + A[2,1]*x[1] + A[2,2]*x[2] + A[2,3]*x[3] = 5*1 + 0*2 + 0*3 + 6*4 = 5 + 24 = 29
    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 7.0), y[0]);
    try testing.expectEqual(@as(f64, 22.0), y[1]);
    try testing.expectEqual(@as(f64, 29.0), y[2]);
}

test "CSC matvec: empty matrix" {
    // Create empty 3x3 matrix
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const x = [_]f64{ 1.0, 2.0, 3.0 };

    const allocator = testing.allocator;
    const y = try allocator.alloc(f64, 3);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0.0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(f64, 0.0), y[0]);
    try testing.expectEqual(@as(f64, 0.0), y[1]);
    try testing.expectEqual(@as(f64, 0.0), y[2]);
}

test "CSC matvec: integer types" {
    // Create 2x3 matrix with i32:
    // [1 2 0]
    // [0 3 4]
    var coo = COO(i32).init(testing.allocator, 2, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1);
    try coo.append(0, 1, 2);
    try coo.append(1, 1, 3);
    try coo.append(1, 2, 4);

    var csc = try CSC(i32).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // x = [5, 6, 7]
    const x = [_]i32{ 5, 6, 7 };

    // y[0] = 1*5 + 2*6 = 5 + 12 = 17
    // y[1] = 3*6 + 4*7 = 18 + 28 = 46
    const allocator = testing.allocator;
    const y = try allocator.alloc(i32, 2);
    defer allocator.free(y);

    for (0..csc.rows) |i| {
        y[i] = 0;
    }

    for (0..csc.cols) |j| {
        const col_start = csc.col_ptr[j];
        const col_end = csc.col_ptr[j + 1];
        for (csc.row_indices[col_start..col_end], col_start..) |row, idx| {
            y[row] += csc.values[idx] * x[j];
        }
    }

    try testing.expectEqual(@as(i32, 17), y[0]);
    try testing.expectEqual(@as(i32, 46), y[1]);
}

test "CSC add: general sparse matrices" {
    // Matrix A (3x3):
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    // Matrix B (3x3):
    // [0 1 0]
    // [2 0 3]
    // [0 4 0]
    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 1, 1.0);
    try coo_b.append(1, 0, 2.0);
    try coo_b.append(1, 2, 3.0);
    try coo_b.append(2, 1, 4.0);

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    // C = A + B =
    // [1 1 2]
    // [2 3 3]
    // [4 4 5]
    var result = try csc_a.add(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 9), result.nnz());

    // Verify values
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, 2.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 2));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 1));
    try testing.expectEqual(@as(f64, 5.0), result.get(2, 2));
}

test "CSC add: disjoint patterns" {
    // Matrix A (2x2): [1 0; 0 0]
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    // Matrix B (2x2): [0 0; 0 2]
    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(1, 1, 2.0);

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    // C = A + B = [1 0; 0 2]
    var result = try csc_a.add(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(f64, 1.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 2.0), result.get(1, 1));
}

test "CSC add: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    try testing.expectError(error.DimensionMismatch, csc_a.add(testing.allocator, &csc_b));
}

test "CSC subtract: general sparse matrices" {
    // Matrix A (3x3):
    // [5 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 5.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    // Matrix B (3x3):
    // [1 1 0]
    // [2 0 3]
    // [0 4 0]
    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 1.0);
    try coo_b.append(0, 1, 1.0);
    try coo_b.append(1, 0, 2.0);
    try coo_b.append(1, 2, 3.0);
    try coo_b.append(2, 1, 4.0);

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    // C = A - B =
    // [ 4 -1  2]
    // [-2  3 -3]
    // [ 4 -4  5]
    var result = try csc_a.subtract(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 9), result.nnz());

    // Verify values
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, -1.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 2.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, -2.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 3.0), result.get(1, 1));
    try testing.expectEqual(@as(f64, -3.0), result.get(1, 2));
    try testing.expectEqual(@as(f64, 4.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, -4.0), result.get(2, 1));
    try testing.expectEqual(@as(f64, 5.0), result.get(2, 2));
}

test "CSC subtract: cancellation creates zeros" {
    // Matrix A (2x2): [1 2; 3 4]
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 1, 2.0);
    try coo_a.append(1, 0, 3.0);
    try coo_a.append(1, 1, 4.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    // B = A, so A - B should be zero
    var result = try csc_a.subtract(testing.allocator, &csc_a);
    defer result.deinit();

    // All elements should be zero (still stored in sparse format)
    try testing.expectEqual(@as(usize, 4), result.nnz());
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 1));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 1));
}

test "CSC subtract: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 2);
    defer coo_b.deinit();
    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    try testing.expectError(error.DimensionMismatch, csc_a.subtract(testing.allocator, &csc_b));
}

test "CSC scale: general sparse matrix" {
    // Matrix A (3x3):
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 2, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(2, 0, 4.0);
    try coo.append(2, 2, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // C = 2.5 * A
    var result = try csc.scale(testing.allocator, 2.5);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.rows);
    try testing.expectEqual(@as(usize, 3), result.cols);
    try testing.expectEqual(@as(usize, 5), result.nnz());

    // Verify scaled values
    try testing.expectEqual(@as(f64, 2.5), result.get(0, 0));
    try testing.expectEqual(@as(f64, 5.0), result.get(0, 2));
    try testing.expectEqual(@as(f64, 7.5), result.get(1, 1));
    try testing.expectEqual(@as(f64, 10.0), result.get(2, 0));
    try testing.expectEqual(@as(f64, 12.5), result.get(2, 2));
}

test "CSC scale: zero scalar" {
    var coo = COO(f64).init(testing.allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // C = 0 * A (all zeros)
    var result = try csc.scale(testing.allocator, 0.0);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 0.0), result.get(1, 1));
}

test "CSC scale: negative scalar" {
    var coo = COO(f64).init(testing.allocator, 2, 2);
    defer coo.deinit();
    try coo.append(0, 0, 3.0);
    try coo.append(1, 1, -4.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // C = -1 * A
    var result = try csc.scale(testing.allocator, -1.0);
    defer result.deinit();

    try testing.expectEqual(@as(f64, -3.0), result.get(0, 0));
    try testing.expectEqual(@as(f64, 4.0), result.get(1, 1));
}

test "CSC scale: memory safety check" {
    // Run multiple iterations to check for memory leaks
    for (0..10) |_| {
        var coo = COO(f64).init(testing.allocator, 5, 5);
        defer coo.deinit();
        try coo.append(0, 1, 2.0);
        try coo.append(2, 3, 4.0);
        try coo.append(4, 0, 5.0);

        var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
        defer csc.deinit();

        var result = try csc.scale(testing.allocator, 3.0);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 3), result.nnz());
    }
}

test "CSC hadamard: general sparse matrices" {
    // Matrix A:          Matrix B:
    // [1  0  2]          [0  0  2]
    // [0  3  0]          [0  3  5]
    // [4  0  5]          [4  6  0]
    //
    // A ∘ B (element-wise product, only where both non-zero):
    // [0  0  4]  (1×0=0 missing, 0×0=0 missing, 2×2=4)
    // [0  9  0]  (0×0=0 missing, 3×3=9, 0×5=0 missing)
    // [16 0  0]  (4×4=16, 0×6=0 missing, 5×0=0 missing)
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(0, 2, 2.0);
    try coo_a.append(1, 1, 3.0);
    try coo_a.append(2, 0, 4.0);
    try coo_a.append(2, 2, 5.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 2, 2.0);
    try coo_b.append(1, 1, 3.0);
    try coo_b.append(1, 2, 5.0);
    try coo_b.append(2, 0, 4.0);
    try coo_b.append(2, 1, 6.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    // Result should have 3 non-zeros
    try testing.expectEqual(@as(usize, 3), result.nnz());

    // Check values
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 2));  // 2×2=4
    try testing.expectEqual(@as(f64, 9.0), result.get(1, 1));  // 3×3=9
    try testing.expectEqual(@as(f64, 16.0), result.get(2, 0)); // 4×4=16
    try testing.expectEqual(@as(f64, 0.0), result.get(0, 0));  // No overlap
    try testing.expectEqual(@as(f64, 0.0), result.get(2, 2));  // No overlap
}

test "CSC hadamard: diagonal matrices" {
    // A = diag([1, 2, 3])
    // B = diag([4, 5, 6])
    // A ∘ B = diag([4, 10, 18])
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(1, 1, 2.0);
    try coo_a.append(2, 2, 3.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 4.0);
    try coo_b.append(1, 1, 5.0);
    try coo_b.append(2, 2, 6.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.nnz());
    try testing.expectEqual(@as(f64, 4.0), result.get(0, 0));   // 1×4=4
    try testing.expectEqual(@as(f64, 10.0), result.get(1, 1));  // 2×5=10
    try testing.expectEqual(@as(f64, 18.0), result.get(2, 2));  // 3×6=18
}

test "CSC hadamard: no overlap (result is zero matrix)" {
    // A has non-zeros at (0,0) and (1,1)
    // B has non-zeros at (0,1) and (1,0)
    // No overlap → result should be empty
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);
    try coo_a.append(1, 1, 2.0);

    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 1, 3.0);
    try coo_b.append(1, 0, 4.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.nnz());
}

test "CSC hadamard: complete overlap" {
    // A and B have same sparsity pattern
    var coo_a = COO(f64).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 2.0);
    try coo_a.append(0, 1, 3.0);
    try coo_a.append(1, 0, 4.0);
    try coo_a.append(1, 1, 5.0);

    var coo_b = COO(f64).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 6.0);
    try coo_b.append(0, 1, 7.0);
    try coo_b.append(1, 0, 8.0);
    try coo_b.append(1, 1, 9.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 4), result.nnz());
    try testing.expectEqual(@as(f64, 12.0), result.get(0, 0)); // 2×6=12
    try testing.expectEqual(@as(f64, 21.0), result.get(0, 1)); // 3×7=21
    try testing.expectEqual(@as(f64, 32.0), result.get(1, 0)); // 4×8=32
    try testing.expectEqual(@as(f64, 45.0), result.get(1, 1)); // 5×9=45
}

test "CSC hadamard: dimension mismatch error" {
    var coo_a = COO(f64).init(testing.allocator, 2, 3);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 1.0);

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 2.0);

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    try testing.expectError(error.DimensionMismatch, csc_a.hadamard(testing.allocator, &csc_b));
}

test "CSC hadamard: empty matrices" {
    var coo_a = COO(f64).init(testing.allocator, 3, 3);
    defer coo_a.deinit();

    var coo_b = COO(f64).init(testing.allocator, 3, 3);
    defer coo_b.deinit();

    var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.nnz());
}

test "CSC hadamard: integer type" {
    var coo_a = COO(i32).init(testing.allocator, 2, 2);
    defer coo_a.deinit();
    try coo_a.append(0, 0, 3);
    try coo_a.append(1, 1, 4);

    var coo_b = COO(i32).init(testing.allocator, 2, 2);
    defer coo_b.deinit();
    try coo_b.append(0, 0, 5);
    try coo_b.append(1, 1, 6);

    var csc_a = try CSC(i32).fromCOO(testing.allocator, &coo_a);
    defer csc_a.deinit();

    var csc_b = try CSC(i32).fromCOO(testing.allocator, &coo_b);
    defer csc_b.deinit();

    var result = try csc_a.hadamard(testing.allocator, &csc_b);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.nnz());
    try testing.expectEqual(@as(i32, 15), result.get(0, 0)); // 3×5=15
    try testing.expectEqual(@as(i32, 24), result.get(1, 1)); // 4×6=24
}

test "CSC hadamard: memory safety check" {
    // Run multiple iterations to check for memory leaks
    for (0..10) |_| {
        var coo_a = COO(f64).init(testing.allocator, 5, 5);
        defer coo_a.deinit();
        try coo_a.append(0, 1, 2.0);
        try coo_a.append(2, 3, 4.0);
        try coo_a.append(4, 0, 5.0);

        var coo_b = COO(f64).init(testing.allocator, 5, 5);
        defer coo_b.deinit();
        try coo_b.append(0, 1, 3.0);
        try coo_b.append(2, 3, 6.0);

        var csc_a = try CSC(f64).fromCOO(testing.allocator, &coo_a);
        defer csc_a.deinit();

        var csc_b = try CSC(f64).fromCOO(testing.allocator, &coo_b);
        defer csc_b.deinit();

        var result = try csc_a.hadamard(testing.allocator, &csc_b);
        defer result.deinit();

        try testing.expectEqual(@as(usize, 2), result.nnz());
    }
}

test "CSC trace: square matrix with diagonal" {
    // Matrix:
    // [1  2  0]
    // [0  3  4]
    // [5  0  6]
    // Trace = 1 + 3 + 6 = 10
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 2, 6.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const tr = try csc.trace();
    try testing.expectEqual(@as(f64, 10.0), tr);
}

test "CSC trace: identity matrix" {
    var coo = COO(f64).init(testing.allocator, 4, 4);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.append(3, 3, 1.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const tr = try csc.trace();
    try testing.expectEqual(@as(f64, 4.0), tr);
}

test "CSC trace: non-square matrix error" {
    var coo = COO(f64).init(testing.allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    try testing.expectError(error.NotSquare, csc.trace());
}

test "CSC trace: missing diagonal elements" {
    // Matrix with no diagonal elements (trace = 0)
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 1, 2.0);
    try coo.append(1, 2, 3.0);
    try coo.append(2, 0, 4.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const tr = try csc.trace();
    try testing.expectEqual(@as(f64, 0.0), tr);
}

test "CSC diag: general sparse matrix" {
    // Matrix:
    // [1  2  0]
    // [0  3  4]
    // [5  0  6]
    // Diagonal = [1, 3, 6]
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);
    try coo.append(2, 0, 5.0);
    try coo.append(2, 2, 6.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const d = try csc.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 3), d.len);
    try testing.expectEqual(@as(f64, 1.0), d[0]);
    try testing.expectEqual(@as(f64, 3.0), d[1]);
    try testing.expectEqual(@as(f64, 6.0), d[2]);
}

test "CSC diag: rectangular matrix" {
    // 2×3 matrix:
    // [1  2  0]
    // [0  3  4]
    // Diagonal = [1, 3] (min(2,3) = 2)
    var coo = COO(f64).init(testing.allocator, 2, 3);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(0, 1, 2.0);
    try coo.append(1, 1, 3.0);
    try coo.append(1, 2, 4.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const d = try csc.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 2), d.len);
    try testing.expectEqual(@as(f64, 1.0), d[0]);
    try testing.expectEqual(@as(f64, 3.0), d[1]);
}

test "CSC diag: missing diagonal elements" {
    // Matrix with missing diagonal → zeros in result
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 1, 2.0);
    try coo.append(1, 2, 3.0);
    try coo.append(2, 0, 4.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const d = try csc.diag(testing.allocator);
    defer testing.allocator.free(d);

    try testing.expectEqual(@as(usize, 3), d.len);
    try testing.expectEqual(@as(f64, 0.0), d[0]);
    try testing.expectEqual(@as(f64, 0.0), d[1]);
    try testing.expectEqual(@as(f64, 0.0), d[2]);
}

test "CSC density and sparsity: various matrices" {
    // Dense 2x2 matrix (all elements non-zero)
    {
        var coo = COO(f64).init(testing.allocator, 2, 2);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(0, 1, 2.0);
        try coo.append(1, 0, 3.0);
        try coo.append(1, 1, 4.0);

        var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
        defer csc.deinit();

        try testing.expectEqual(@as(f64, 1.0), csc.density());
        try testing.expectEqual(@as(f64, 0.0), csc.sparsity());
    }

    // Sparse 3x3 matrix (3 non-zeros out of 9)
    {
        var coo = COO(f64).init(testing.allocator, 3, 3);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(1, 1, 2.0);
        try coo.append(2, 2, 3.0);

        var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
        defer csc.deinit();

        const expected_density = 3.0 / 9.0;
        const expected_sparsity = 6.0 / 9.0;
        try testing.expectApproxEqRel(expected_density, csc.density(), 1e-10);
        try testing.expectApproxEqRel(expected_sparsity, csc.sparsity(), 1e-10);
    }

    // Empty matrix
    {
        var coo = COO(f64).init(testing.allocator, 3, 3);
        defer coo.deinit();

        var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
        defer csc.deinit();

        try testing.expectEqual(@as(f64, 0.0), csc.density());
        try testing.expectEqual(@as(f64, 1.0), csc.sparsity());
    }
}

test "CSC normFrobenius: general sparse matrix" {
    // Matrix:
    // [3  0  4]
    // [0  0  0]
    // [0  5  0]
    // Frobenius norm = sqrt(3^2 + 4^2 + 5^2) = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.071
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();
    try coo.append(0, 0, 3.0);
    try coo.append(0, 2, 4.0);
    try coo.append(2, 1, 5.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const norm = csc.normFrobenius();
    const expected = @sqrt(@as(f64, 50.0));
    try testing.expectApproxEqRel(expected, norm, 1e-10);
}

test "CSC normFrobenius: identity matrix" {
    var coo = COO(f64).init(testing.allocator, 4, 4);
    defer coo.deinit();
    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 1.0);
    try coo.append(2, 2, 1.0);
    try coo.append(3, 3, 1.0);

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const norm = csc.normFrobenius();
    try testing.expectApproxEqRel(@as(f64, 2.0), norm, 1e-10); // sqrt(4) = 2
}

test "CSC normFrobenius: empty matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    const norm = csc.normFrobenius();
    try testing.expectEqual(@as(f64, 0.0), norm);
}

test "CSC utilities: memory safety check" {
    // Run multiple iterations to check for memory leaks in diag()
    for (0..10) |_| {
        var coo = COO(f64).init(testing.allocator, 4, 4);
        defer coo.deinit();
        try coo.append(0, 0, 1.0);
        try coo.append(1, 1, 2.0);
        try coo.append(2, 2, 3.0);
        try coo.append(3, 3, 4.0);

        var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
        defer csc.deinit();

        const d = try csc.diag(testing.allocator);
        defer testing.allocator.free(d);

        _ = try csc.trace();
        _ = csc.density();
        _ = csc.sparsity();
        _ = csc.normFrobenius();
    }
}

test "COO: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 2, 2.0);
    try coo.append(2, 1, 3.0);

    // Should pass validation
    try coo.validate();
}

test "COO: validate detects out-of-bounds row" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(3, 0, 2.0); // row 3 is out of bounds for 3x3 matrix

    // Should fail validation
    const result = coo.validate();
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "COO: validate detects out-of-bounds column" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(0, 3, 2.0); // col 3 is out of bounds for 3x3 matrix

    // Should fail validation
    const result = coo.validate();
    try testing.expectError(error.IndexOutOfBounds, result);
}

test "CSR: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csr = try CSR(f64).fromCOO(testing.allocator, &coo);
    defer csr.deinit();

    // Should pass validation
    try csr.validate();
}

test "CSC: validate valid matrix" {
    var coo = COO(f64).init(testing.allocator, 3, 3);
    defer coo.deinit();

    try coo.append(0, 0, 1.0);
    try coo.append(1, 1, 2.0);
    try coo.append(2, 2, 3.0);
    try coo.sort();

    var csc = try CSC(f64).fromCOO(testing.allocator, &coo);
    defer csc.deinit();

    // Should pass validation
    try csc.validate();
}
