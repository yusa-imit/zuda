/// Matrix Operations Demo - BLAS & Linear Algebra Examples
///
/// This example demonstrates common matrix operations using zuda's BLAS
/// and linear algebra interfaces:
/// - Creating matrices and vectors
/// - BLAS Level 1: Vector dot products and norms
/// - BLAS Level 2: Matrix-vector multiplication (GEMV)
/// - BLAS Level 3: Matrix-matrix multiplication (GEMM)
/// - Solving linear systems
///
/// Run with: zig build example-matrix
const std = @import("std");
const zuda = @import("zuda");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Matrix Operations Demo ===\n\n", .{});

    // 1. Creating vectors and matrices
    std.debug.print("1. Creating Matrices and Vectors\n", .{});
    std.debug.print("---------------------------------\n", .{});

    // Create vectors for BLAS Level 1 operations
    const x_data = [_]f64{ 1, 2, 3 };
    var x = try zuda.ndarray.NDArray(f64, 1).fromSlice(
        allocator,
        &[_]usize{3},
        &x_data,
        .row_major,
    );
    defer x.deinit();

    const y_data = [_]f64{ 4, 5, 6 };
    var y = try zuda.ndarray.NDArray(f64, 1).fromSlice(
        allocator,
        &[_]usize{3},
        &y_data,
        .row_major,
    );
    defer y.deinit();

    std.debug.print("Vector x: ", .{});
    printVector(f64, x);
    std.debug.print("Vector y: ", .{});
    printVector(f64, y);

    // 2. BLAS Level 1: Vector operations
    std.debug.print("\n2. BLAS Level 1: Vector Operations\n", .{});
    std.debug.print("-----------------------------------\n", .{});

    // Dot product
    const dot_result = try zuda.linalg.blas.dot(f64, x, y);
    std.debug.print("x · y = {d:.2} (expected: 1*4 + 2*5 + 3*6 = 32)\n", .{dot_result});

    // L2 norm
    const norm_x = try zuda.linalg.blas.nrm2(f64, x);
    std.debug.print("||x||₂ = {d:.4} (expected: sqrt(1+4+9) = 3.7417)\n", .{norm_x});

    // AXPY: y := alpha*x + y
    var y_axpy = try zuda.ndarray.NDArray(f64, 1).fromSlice(
        allocator,
        &[_]usize{3},
        &y_data,
        .row_major,
    );
    defer y_axpy.deinit();
    try zuda.linalg.blas.axpy(f64, 2.0, x, &y_axpy); // y := 2*x + y
    std.debug.print("2*x + y = ", .{});
    printVector(f64, y_axpy);
    std.debug.print("         (expected: [2+4, 4+5, 6+6] = [6, 9, 12])\n", .{});

    // 3. BLAS Level 2: Matrix-vector multiplication
    std.debug.print("\n3. BLAS Level 2: Matrix-Vector (GEMV)\n", .{});
    std.debug.print("--------------------------------------\n", .{});

    // Create a 2x3 matrix:
    // [ 1  2  3 ]
    // [ 4  5  6 ]
    const m_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var M = try zuda.ndarray.NDArray(f64, 2).fromSlice(
        allocator,
        &[_]usize{ 2, 3 },
        &m_data,
        .row_major,
    );
    defer M.deinit();

    // Vector v = [1, 2, 3]
    const v_data = [_]f64{ 1, 2, 3 };
    var v = try zuda.ndarray.NDArray(f64, 1).fromSlice(
        allocator,
        &[_]usize{3},
        &v_data,
        .row_major,
    );
    defer v.deinit();

    // Result vector (2 elements)
    var result = try zuda.ndarray.NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer result.deinit();

    std.debug.print("Matrix M (2x3):\n", .{});
    printMatrix(f64, M);
    std.debug.print("Vector v: ", .{});
    printVector(f64, v);

    // Compute M * v using BLAS gemv: y = α*M*v + β*y
    try zuda.linalg.blas.gemv(f64, 1.0, M, v, 0.0, &result);
    std.debug.print("M * v = ", .{});
    printVector(f64, result);
    std.debug.print("       (expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32])\n", .{});

    // 4. BLAS Level 3: Matrix-matrix multiplication
    std.debug.print("\n4. BLAS Level 3: Matrix-Matrix (GEMM)\n", .{});
    std.debug.print("--------------------------------------\n", .{});

    // Create two 2x2 matrices:
    // A = [ 1  2 ]    B = [ 5  6 ]
    //     [ 3  4 ]        [ 7  8 ]
    const a_data = [_]f64{ 1, 2, 3, 4 };
    var A = try zuda.ndarray.NDArray(f64, 2).fromSlice(
        allocator,
        &[_]usize{ 2, 2 },
        &a_data,
        .row_major,
    );
    defer A.deinit();

    const b_data = [_]f64{ 5, 6, 7, 8 };
    var B = try zuda.ndarray.NDArray(f64, 2).fromSlice(
        allocator,
        &[_]usize{ 2, 2 },
        &b_data,
        .row_major,
    );
    defer B.deinit();

    // Result matrix C (2x2)
    var C = try zuda.ndarray.NDArray(f64, 2).zeros(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer C.deinit();

    std.debug.print("Matrix A (2x2):\n", .{});
    printMatrix(f64, A);
    std.debug.print("Matrix B (2x2):\n", .{});
    printMatrix(f64, B);

    // Compute C = α*A*B + β*C using BLAS gemm
    try zuda.linalg.blas.gemm(f64, 1.0, A, B, 0.0, &C);
    std.debug.print("A * B =\n", .{});
    printMatrix(f64, C);
    std.debug.print("Expected:\n", .{});
    std.debug.print("  [ 1*5+2*7   1*6+2*8 ]   [ 19  22 ]\n", .{});
    std.debug.print("  [ 3*5+4*7   3*6+4*8 ] = [ 43  50 ]\n", .{});

    // 5. Solving linear systems
    std.debug.print("\n5. Solving Linear Systems\n", .{});
    std.debug.print("-------------------------\n", .{});

    // Solve Ax = b for the system:
    // 2x + y = 5
    // x + 3y = 11
    // Solution: x = 1, y = 3
    const a_sys_data = [_]f64{ 2, 1, 1, 3 };
    var A_sys = try zuda.ndarray.NDArray(f64, 2).fromSlice(
        allocator,
        &[_]usize{ 2, 2 },
        &a_sys_data,
        .row_major,
    );
    defer A_sys.deinit();

    const b_sys_data = [_]f64{ 5, 11 };
    var b_sys = try zuda.ndarray.NDArray(f64, 1).fromSlice(
        allocator,
        &[_]usize{2},
        &b_sys_data,
        .row_major,
    );
    defer b_sys.deinit();

    std.debug.print("System of equations:\n", .{});
    std.debug.print("  2x + y  = 5\n", .{});
    std.debug.print("  x  + 3y = 11\n", .{});

    var x_sol = try zuda.linalg.solve.solve(f64, A_sys, b_sys, allocator);
    defer x_sol.deinit();

    std.debug.print("Solution x: ", .{});
    printVector(f64, x_sol);
    std.debug.print("           (expected: [1.0, 3.0])\n", .{});

    // Verify: A * x should equal b
    var b_verify = try zuda.ndarray.NDArray(f64, 1).zeros(allocator, &[_]usize{2}, .row_major);
    defer b_verify.deinit();
    try zuda.linalg.blas.gemv(f64, 1.0, A_sys, x_sol, 0.0, &b_verify);
    std.debug.print("Verification (A*x): ", .{});
    printVector(f64, b_verify);
    std.debug.print("                   (should match b = [5.0, 11.0])\n", .{});

    std.debug.print("\n=== Demo Complete ===\n\n", .{});
}

/// Helper function to print a 2D matrix
fn printMatrix(comptime T: type, matrix: zuda.ndarray.NDArray(T, 2)) void {
    const rows = matrix.shape[0];
    const cols = matrix.shape[1];

    for (0..rows) |i| {
        std.debug.print("  [ ", .{});
        for (0..cols) |j| {
            const indices = [_]isize{ @intCast(i), @intCast(j) };
            const val = matrix.get(&indices) catch unreachable;
            std.debug.print("{d:6.2} ", .{val});
        }
        std.debug.print("]\n", .{});
    }
}

/// Helper function to print a 1D vector
fn printVector(comptime T: type, vector: zuda.ndarray.NDArray(T, 1)) void {
    const len = vector.shape[0];
    std.debug.print("[ ", .{});
    for (0..len) |i| {
        const indices = [_]isize{@intCast(i)};
        const val = vector.get(&indices) catch unreachable;
        std.debug.print("{d:.2} ", .{val});
    }
    std.debug.print("]\n", .{});
}
