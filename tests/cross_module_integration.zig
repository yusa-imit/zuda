//! Cross-module Integration Tests (Phase 12 - v1.28.0)
//!
//! Verifies seamless interoperability between zuda's v2.0 scientific computing modules.
//! These tests demonstrate that NDArray ↔ linalg works naturally with shared types.

const std = @import("std");
const testing = std.testing;
const zuda = @import("zuda");
const NDArray = zuda.ndarray.NDArray;

test "cross-module: NDArray → linalg SVD → NDArray results" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    @memcpy(A.data, &data);

    // linalg.decompositions.svd takes NDArray, returns NDArrays!
    var result = try zuda.linalg.decompositions.svd(f64, A, allocator);
    defer result.U.deinit();
    defer result.S.deinit();
    defer result.Vt.deinit();

    // Verify singular values sorted descending
    try testing.expect(result.S.data[0] >= result.S.data[1]);
}

test "cross-module: NDArray → linalg QR → NDArray results" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 3, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 1, 0, 0, 1, 0, 0 };
    @memcpy(A.data, &data);

    var result = try zuda.linalg.decompositions.qr(f64, A, allocator);
    defer result.Q.deinit();
    defer result.R.deinit();

    try testing.expectEqual(@as(usize, 3), result.Q.shape[0]);
    try testing.expectEqual(@as(usize, 3), result.Q.shape[1]);
}

test "cross-module: NDArray → linalg Cholesky → NDArray result" {
    const allocator = testing.allocator;

    var A = try NDArray(f64, 2).init(allocator, &[_]usize{ 2, 2 }, .row_major);
    defer A.deinit();

    const data = [_]f64{ 4, 2, 2, 3 };
    @memcpy(A.data, &data);

    var L = try zuda.linalg.decompositions.cholesky(f64, A, allocator);
    defer L.deinit();

    try testing.expectEqual(@as(usize, 2), L.shape[0]);
    try testing.expectEqual(@as(usize, 2), L.shape[1]);
}
