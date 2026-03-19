/// Utility functions for common operations with zuda containers.
///
/// This module provides convenience functions to reduce boilerplate when
/// working with containers. Instead of writing custom comparators and hash
/// functions, you can use the pre-built utilities here.
///
/// Example:
/// ```zig
/// const zuda = @import("zuda");
///
/// // Instead of writing a custom comparator:
/// var tree = try zuda.containers.trees.RedBlackTree(
///     i64, []const u8, void, zuda.utils.compare.ascending(i64)
/// ).init(allocator, {});
///
/// // Instead of writing a custom hash function:
/// var map = try zuda.containers.hashing.RobinHoodHashMap(
///     Point, i32, void, zuda.utils.hash.auto(Point)
/// ).init(allocator, {});
/// ```

pub const compare = @import("utils/compare.zig");
pub const hash = @import("utils/hash.zig");
pub const builder = @import("utils/builder.zig");
pub const debug = @import("utils/debug.zig");
pub const perf = @import("utils/perf.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
