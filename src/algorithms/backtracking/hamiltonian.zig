/// Hamiltonian Path and Cycle algorithms using backtracking.
///
/// A Hamiltonian path is a path that visits each vertex exactly once.
/// A Hamiltonian cycle is a Hamiltonian path that returns to the starting vertex.
///
/// ## Algorithm
///
/// Backtracking with DFS exploration:
/// 1. Start from a vertex (or try all vertices for path)
/// 2. Mark vertex as visited and add to path
/// 3. Try all unvisited neighbors recursively
/// 4. If all vertices visited:
///    - For path: success
///    - For cycle: check if edge back to start exists
/// 5. Backtrack: unmark vertex and remove from path
///
/// ## Time Complexity
///
/// - hamiltonianPath(): O(N!) worst case (try all permutations)
/// - hamiltonianCycle(): O(N!) worst case
/// - With pruning: significantly better for sparse graphs
///
/// ## Space Complexity
///
/// O(N) for recursion stack and visited tracking
///
/// ## Use Cases
///
/// - Graph theory (Hamiltonian path problem - NP-complete)
/// - Routing problems (visit all cities exactly once)
/// - Circuit design (tracing paths through components)
/// - Bioinformatics (genome sequencing - de Bruijn graphs)
/// - Game theory (chess knight's tour is Hamiltonian path variant)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of Hamiltonian path search
pub fn PathResult(comptime T: type) type {
    return struct {
        path: []T,
        found: bool,

        pub fn deinit(self: *@This(), allocator: Allocator) void {
            if (self.found) {
                allocator.free(self.path);
            }
        }
    };
}

/// Find a Hamiltonian path in the graph starting from the given vertex.
/// Returns the path if found, otherwise returns .{ .path = undefined, .found = false }.
///
/// Time: O(N!) worst case
/// Space: O(N) for recursion stack + path storage
///
/// ## Parameters
/// - `T`: Vertex type (must be hashable and comparable)
/// - `allocator`: Memory allocator
/// - `graph`: Adjacency list representation (HashMap(T, ArrayList(T)))
/// - `start`: Starting vertex
///
/// ## Example
/// ```zig
/// var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
/// // ... populate graph ...
/// var result = try hamiltonianPath(u32, allocator, &graph, 0);
/// defer result.deinit(allocator);
/// if (result.found) {
///     // Use result.path
/// }
/// ```
pub fn hamiltonianPath(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    start: T,
) !PathResult(T) {
    if (!graph.contains(start)) {
        return error.InvalidStartVertex;
    }

    const n = graph.count();
    if (n == 0) {
        return .{ .path = undefined, .found = false };
    }

    var visited = std.AutoHashMap(T, bool).init(allocator);
    defer visited.deinit();

    var path = std.ArrayList(T){};
    defer path.deinit(allocator);

    // Initialize visited map
    var it = graph.keyIterator();
    while (it.next()) |vertex| {
        try visited.put(vertex.*, false);
    }

    // Try to find path using backtracking
    const found = try hamiltonianPathBacktrack(T, allocator, graph, start, &visited, &path, n);

    if (found) {
        return .{
            .path = try path.toOwnedSlice(allocator),
            .found = true,
        };
    } else {
        return .{ .path = undefined, .found = false };
    }
}

/// Find a Hamiltonian cycle in the graph starting from the given vertex.
/// A Hamiltonian cycle visits all vertices exactly once and returns to start.
///
/// Time: O(N!) worst case
/// Space: O(N) for recursion stack + path storage
pub fn hamiltonianCycle(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    start: T,
) !PathResult(T) {
    if (!graph.contains(start)) {
        return error.InvalidStartVertex;
    }

    const n = graph.count();
    if (n < 3) {
        // Need at least 3 vertices for a cycle
        return .{ .path = undefined, .found = false };
    }

    var visited = std.AutoHashMap(T, bool).init(allocator);
    defer visited.deinit();

    var path = std.ArrayList(T){};
    defer path.deinit(allocator);

    // Initialize visited map
    var it = graph.keyIterator();
    while (it.next()) |vertex| {
        try visited.put(vertex.*, false);
    }

    // Try to find cycle using backtracking
    const found = try hamiltonianCycleBacktrack(T, allocator, graph, start, start, &visited, &path, n);

    if (found) {
        return .{
            .path = try path.toOwnedSlice(allocator),
            .found = true,
        };
    } else {
        return .{ .path = undefined, .found = false };
    }
}

/// Check if a given path is a valid Hamiltonian path in the graph.
/// Validates: correct length, all vertices unique, all edges exist
pub fn isValidPath(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    path: []const T,
) !bool {
    const n = graph.count();

    // Check length
    if (path.len != n) {
        return false;
    }

    // Check uniqueness
    var seen = std.AutoHashMap(T, bool).init(allocator);
    defer seen.deinit();

    for (path) |vertex| {
        if (!graph.contains(vertex)) {
            return false; // Vertex not in graph
        }
        if (seen.contains(vertex)) {
            return false; // Duplicate vertex
        }
        try seen.put(vertex, true);
    }

    // Check edges
    for (path[0 .. path.len - 1], 0..) |vertex, i| {
        const neighbors = graph.get(vertex) orelse return false;
        const next = path[i + 1];

        // Check if edge exists
        var edge_exists = false;
        for (neighbors.items) |neighbor| {
            if (std.meta.eql(neighbor, next)) {
                edge_exists = true;
                break;
            }
        }
        if (!edge_exists) {
            return false;
        }
    }

    return true;
}

/// Check if a given path is a valid Hamiltonian cycle in the graph.
/// Validates: valid path + edge from last vertex back to first
pub fn isValidCycle(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    path: []const T,
) !bool {
    if (path.len < 3) {
        return false; // Need at least 3 vertices for a cycle
    }

    // First check if it's a valid path
    if (!try isValidPath(T, allocator, graph, path)) {
        return false;
    }

    // Check edge from last vertex back to first
    const last = path[path.len - 1];
    const first = path[0];
    const neighbors = graph.get(last) orelse return false;

    for (neighbors.items) |neighbor| {
        if (std.meta.eql(neighbor, first)) {
            return true;
        }
    }

    return false;
}

// --- Helper Functions ---

fn hamiltonianPathBacktrack(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    current: T,
    visited: *std.AutoHashMap(T, bool),
    path: *std.ArrayList(T),
    n: usize,
) !bool {
    // Mark current as visited and add to path
    try visited.put(current, true);
    try path.append(allocator, current);

    // Base case: all vertices visited
    if (path.items.len == n) {
        return true;
    }

    // Try all unvisited neighbors
    const neighbors = graph.get(current) orelse return false;
    for (neighbors.items) |neighbor| {
        if (!(visited.get(neighbor) orelse false)) {
            if (try hamiltonianPathBacktrack(T, allocator, graph, neighbor, visited, path, n)) {
                return true;
            }
        }
    }

    // Backtrack: unmark and remove from path
    try visited.put(current, false);
    _ = path.pop();
    return false;
}

fn hamiltonianCycleBacktrack(
    comptime T: type,
    allocator: Allocator,
    graph: *const std.AutoHashMap(T, std.ArrayList(T)),
    start: T,
    current: T,
    visited: *std.AutoHashMap(T, bool),
    path: *std.ArrayList(T),
    n: usize,
) !bool {
    // Mark current as visited and add to path
    try visited.put(current, true);
    try path.append(allocator, current);

    // Base case: all vertices visited
    if (path.items.len == n) {
        // Check if there's an edge back to start
        const neighbors = graph.get(current) orelse return false;
        for (neighbors.items) |neighbor| {
            if (std.meta.eql(neighbor, start)) {
                return true;
            }
        }
        // No edge back to start - backtrack
        try visited.put(current, false);
        _ = path.pop();
        return false;
    }

    // Try all unvisited neighbors
    const neighbors = graph.get(current) orelse return false;
    for (neighbors.items) |neighbor| {
        if (!(visited.get(neighbor) orelse false)) {
            if (try hamiltonianCycleBacktrack(T, allocator, graph, start, neighbor, visited, path, n)) {
                return true;
            }
        }
    }

    // Backtrack: unmark and remove from path
    try visited.put(current, false);
    _ = path.pop();
    return false;
}

// --- Tests ---

test "hamiltonianPath - simple path" {
    const allocator = std.testing.allocator;

    // Graph: 0 -> 1 -> 2 -> 3
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 3);
    try graph.put(2, neighbors2);

    const neighbors3 = std.ArrayList(u32){};
    try graph.put(3, neighbors3);

    var result = try hamiltonianPath(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 4), result.path.len);
    try std.testing.expectEqual(@as(u32, 0), result.path[0]);
    try std.testing.expectEqual(@as(u32, 1), result.path[1]);
    try std.testing.expectEqual(@as(u32, 2), result.path[2]);
    try std.testing.expectEqual(@as(u32, 3), result.path[3]);
}

test "hamiltonianPath - complete graph K4" {
    const allocator = std.testing.allocator;

    // Complete graph with 4 vertices (all connected)
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    for (0..4) |i| {
        var neighbors = std.ArrayList(u32){};
        for (0..4) |j| {
            if (i != j) {
                try neighbors.append(allocator, @intCast(j));
            }
        }
        try graph.put(@intCast(i), neighbors);
    }

    var result = try hamiltonianPath(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 4), result.path.len);

    // Validate it's a valid path
    try std.testing.expect(try isValidPath(u32, allocator, &graph, result.path));
}

test "hamiltonianPath - no path exists" {
    const allocator = std.testing.allocator;

    // Disconnected graph: 0-1  2-3
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 3);
    try graph.put(2, neighbors2);

    var neighbors3 = std.ArrayList(u32){};
    try neighbors3.append(allocator, 2);
    try graph.put(3, neighbors3);

    var result = try hamiltonianPath(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(!result.found);
}

test "hamiltonianCycle - simple cycle" {
    const allocator = std.testing.allocator;

    // Triangle: 0 <-> 1 <-> 2 <-> 0
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try neighbors0.append(allocator, 2);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 1);
    try neighbors2.append(allocator, 0);
    try graph.put(2, neighbors2);

    var result = try hamiltonianCycle(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 3), result.path.len);

    // Validate it's a valid cycle
    try std.testing.expect(try isValidCycle(u32, allocator, &graph, result.path));
}

test "hamiltonianCycle - square graph" {
    const allocator = std.testing.allocator;

    // Square: 0-1-2-3-0
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try neighbors0.append(allocator, 3);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 1);
    try neighbors2.append(allocator, 3);
    try graph.put(2, neighbors2);

    var neighbors3 = std.ArrayList(u32){};
    try neighbors3.append(allocator, 2);
    try neighbors3.append(allocator, 0);
    try graph.put(3, neighbors3);

    var result = try hamiltonianCycle(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 4), result.path.len);

    // Validate it's a valid cycle
    try std.testing.expect(try isValidCycle(u32, allocator, &graph, result.path));
}

test "hamiltonianCycle - no cycle exists" {
    const allocator = std.testing.allocator;

    // Path graph: 0-1-2-3 (no edge 3-0)
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 1);
    try neighbors2.append(allocator, 3);
    try graph.put(2, neighbors2);

    var neighbors3 = std.ArrayList(u32){};
    try neighbors3.append(allocator, 2);
    try graph.put(3, neighbors3);

    var result = try hamiltonianCycle(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(!result.found);
}

test "hamiltonianCycle - too few vertices" {
    const allocator = std.testing.allocator;

    // Graph with only 2 vertices
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try graph.put(1, neighbors1);

    var result = try hamiltonianCycle(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(!result.found);
}

test "isValidPath - valid path" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    const neighbors2 = std.ArrayList(u32){};
    try graph.put(2, neighbors2);

    const path = [_]u32{ 0, 1, 2 };
    try std.testing.expect(try isValidPath(u32, allocator, &graph, &path));
}

test "isValidPath - wrong length" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    const neighbors0 = std.ArrayList(u32){};
    try graph.put(0, neighbors0);

    const neighbors1 = std.ArrayList(u32){};
    try graph.put(1, neighbors1);

    const path = [_]u32{0}; // Too short
    try std.testing.expect(!try isValidPath(u32, allocator, &graph, &path));
}

test "isValidPath - duplicate vertices" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 0);
    try graph.put(1, neighbors1);

    const path = [_]u32{ 0, 1, 0 }; // Wrong length anyway, but has duplicate
    try std.testing.expect(!try isValidPath(u32, allocator, &graph, &path));
}

test "isValidPath - missing edge" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    const neighbors0 = std.ArrayList(u32){};
    try graph.put(0, neighbors0);

    const neighbors1 = std.ArrayList(u32){};
    try graph.put(1, neighbors1);

    const neighbors2 = std.ArrayList(u32){};
    try graph.put(2, neighbors2);

    const path = [_]u32{ 0, 1, 2 }; // No edges exist
    try std.testing.expect(!try isValidPath(u32, allocator, &graph, &path));
}

test "isValidCycle - valid cycle" {
    const allocator = std.testing.allocator;

    // Triangle
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try neighbors0.append(allocator, 2);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    var neighbors2 = std.ArrayList(u32){};
    try neighbors2.append(allocator, 0);
    try graph.put(2, neighbors2);

    const path = [_]u32{ 0, 1, 2 };
    try std.testing.expect(try isValidCycle(u32, allocator, &graph, &path));
}

test "isValidCycle - no return edge" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    var neighbors0 = std.ArrayList(u32){};
    try neighbors0.append(allocator, 1);
    try graph.put(0, neighbors0);

    var neighbors1 = std.ArrayList(u32){};
    try neighbors1.append(allocator, 2);
    try graph.put(1, neighbors1);

    const neighbors2 = std.ArrayList(u32){};
    // No edge back to 0
    try graph.put(2, neighbors2);

    const path = [_]u32{ 0, 1, 2 };
    try std.testing.expect(!try isValidCycle(u32, allocator, &graph, &path));
}

test "hamiltonianPath - invalid start vertex" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    const neighbors0 = std.ArrayList(u32){};
    try graph.put(0, neighbors0);

    const result = hamiltonianPath(u32, allocator, &graph, 99);
    try std.testing.expectError(error.InvalidStartVertex, result);
}

test "hamiltonianPath - single vertex" {
    const allocator = std.testing.allocator;

    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    const neighbors0 = std.ArrayList(u32){};
    try graph.put(0, neighbors0);

    var result = try hamiltonianPath(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 1), result.path.len);
    try std.testing.expectEqual(@as(u32, 0), result.path[0]);
}

test "hamiltonianPath - Peterson graph" {
    const allocator = std.testing.allocator;

    // Peterson graph (5 outer + 5 inner vertices)
    // Known to have Hamiltonian paths but no Hamiltonian cycle
    var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
    defer {
        var it = graph.valueIterator();
        while (it.next()) |list| {
            list.deinit(allocator);
        }
        graph.deinit();
    }

    // Outer pentagon (0-4)
    for (0..5) |i| {
        var neighbors = std.ArrayList(u32){};
        const next = @as(u32, @intCast((i + 1) % 5));
        const prev = @as(u32, @intCast((i + 4) % 5));
        const inner = @as(u32, @intCast(i + 5));
        try neighbors.append(allocator, next);
        try neighbors.append(allocator, prev);
        try neighbors.append(allocator, inner);
        try graph.put(@intCast(i), neighbors);
    }

    // Inner pentagram (5-9)
    for (0..5) |i| {
        var neighbors = std.ArrayList(u32){};
        const next = @as(u32, @intCast(5 + ((i + 2) % 5)));
        const prev = @as(u32, @intCast(5 + ((i + 3) % 5)));
        const outer = @as(u32, @intCast(i));
        try neighbors.append(allocator, next);
        try neighbors.append(allocator, prev);
        try neighbors.append(allocator, outer);
        try graph.put(@intCast(5 + i), neighbors);
    }

    var result = try hamiltonianPath(u32, allocator, &graph, 0);
    defer result.deinit(allocator);

    try std.testing.expect(result.found);
    try std.testing.expectEqual(@as(usize, 10), result.path.len);
    try std.testing.expect(try isValidPath(u32, allocator, &graph, result.path));
}

test "hamiltonianPath - memory safety" {
    const allocator = std.testing.allocator;

    // Run multiple iterations to check for leaks
    for (0..5) |_| {
        var graph = std.AutoHashMap(u32, std.ArrayList(u32)).init(allocator);
        defer {
            var it = graph.valueIterator();
            while (it.next()) |list| {
                list.deinit(allocator);
            }
            graph.deinit();
        }

        for (0..4) |i| {
            var neighbors = std.ArrayList(u32){};
            if (i < 3) {
                try neighbors.append(allocator, @intCast(i + 1));
            }
            try graph.put(@intCast(i), neighbors);
        }

        var result = try hamiltonianPath(u32, allocator, &graph, 0);
        defer result.deinit(allocator);

        try std.testing.expect(result.found);
    }
}
