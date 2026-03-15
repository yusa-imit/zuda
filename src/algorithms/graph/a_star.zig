const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A* - Heuristic-guided single-source shortest path algorithm.
///
/// A* is an informed search algorithm that uses a heuristic function to guide
/// the search towards the goal. It guarantees finding the shortest path if the
/// heuristic is admissible (never overestimates the true cost to goal).
///
/// Time Complexity: O(E) in worst case, typically much better with good heuristic
/// Space Complexity: O(V) for distance maps, parent map, and priority queue
///
/// Constraints:
/// - All edge weights must be non-negative
/// - Heuristic function must be admissible (h(v) ≤ actual distance to goal)
/// - For consistency, heuristic should satisfy triangle inequality: h(u) ≤ cost(u,v) + h(v)
///
/// Generic parameters:
/// - V: Vertex type (must be hashable)
/// - W: Weight type (must support comparison and addition)
/// - Context: Context type for hashing/comparing vertices (must have .hash and .eql methods)
pub fn AStar(
    comptime V: type,
    comptime W: type,
    comptime Context: type,
) type {
    return struct {
        const Self = @This();
        const HashMapContext = struct {
            user_ctx: Context,

            pub fn hash(ctx: @This(), key: V) u64 {
                return ctx.user_ctx.hash(key);
            }

            pub fn eql(ctx: @This(), a: V, b: V) bool {
                return ctx.user_ctx.eql(a, b);
            }
        };

        /// Priority queue entry for A*
        /// f = g + h, where g is actual cost from start, h is heuristic to goal
        const QueueEntry = struct {
            vertex: V,
            f_score: W, // g + h (total estimated cost)
        };

        /// Comparison function for min-heap (smallest f-score first)
        fn compareFScore(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
            return std.math.order(a.f_score, b.f_score);
        }

        /// A* result containing shortest path information
        pub const Result = struct {
            /// Actual distance from start vertex to each vertex (g-score)
            distances: std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Parent pointers for path reconstruction
            parents: std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage),
            /// Whether the goal was reached
            goal_reached: bool,
            allocator: Allocator,
            context: Context,

            pub fn deinit(self: *Result) void {
                self.distances.deinit();
                self.parents.deinit();
            }

            /// Get the distance to a vertex from the start vertex.
            /// Returns null if the vertex is not reachable or not yet explored.
            pub fn getDistance(self: *const Result, vertex: V) ?W {
                return self.distances.get(vertex);
            }

            /// Get the parent of a vertex in the shortest path tree.
            /// Returns null if the vertex has no parent (is the start vertex or unreachable).
            pub fn getParent(self: *const Result, vertex: V) ?V {
                return self.parents.get(vertex);
            }

            /// Reconstruct the shortest path from start to target.
            /// Returns null if target is not reachable from start.
            /// Caller owns the returned slice.
            pub fn getPath(self: *const Result, target: V) !?[]V {
                if (!self.distances.contains(target)) {
                    return null;
                }

                var path: std.ArrayList(V) = .{};
                errdefer path.deinit(self.allocator);

                var current = target;
                while (true) {
                    try path.append(self.allocator, current);
                    const parent = self.parents.get(current) orelse break;
                    current = parent;
                }

                // Reverse to get path from start to target
                std.mem.reverse(V, path.items);
                return try path.toOwnedSlice(self.allocator);
            }
        };

        /// Run A* algorithm from start to goal vertex.
        ///
        /// Parameters:
        /// - allocator: Memory allocator
        /// - context: Context for vertex hashing and comparison
        /// - start: Starting vertex
        /// - goal: Goal vertex
        /// - neighbors_fn: Function to get neighbors and edge weights
        ///   Signature: fn (ctx: anytype, vertex: V, out: *std.ArrayList(struct{neighbor: V, weight: W})) anyerror!void
        /// - heuristic_fn: Admissible heuristic function estimating cost from vertex to goal
        ///   Signature: fn (ctx: anytype, vertex: V, goal: V) W
        /// - neighbors_ctx: Context passed to neighbors_fn
        /// - heuristic_ctx: Context passed to heuristic_fn
        ///
        /// Returns: Result struct containing distances, parents, and goal_reached status
        /// Caller owns the returned Result and must call deinit().
        ///
        /// Time: O(E) worst case, typically O(b^d) where b is branching factor, d is depth
        /// Space: O(V)
        pub fn run(
            allocator: Allocator,
            context: Context,
            start: V,
            goal: V,
            neighbors_fn: anytype,
            heuristic_fn: anytype,
            neighbors_ctx: anytype,
            heuristic_ctx: anytype,
        ) !Result {
            // g-score: actual cost from start to vertex
            var g_scores = std.HashMap(V, W, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer g_scores.deinit();

            // Parent map for path reconstruction
            var parents = std.HashMap(V, V, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            errdefer parents.deinit();

            // Priority queue ordered by f-score (g + h)
            var pq = std.PriorityQueue(QueueEntry, void, compareFScore).init(allocator, {});
            defer pq.deinit();

            // Closed set (already processed vertices)
            var closed = std.HashMap(V, void, HashMapContext, std.hash_map.default_max_load_percentage).init(allocator);
            defer closed.deinit();

            // Initialize start vertex
            try g_scores.put(start, @as(W, 0));
            const h_start = heuristic_fn(heuristic_ctx, start, goal);
            try pq.add(.{ .vertex = start, .f_score = h_start });

            var goal_reached = false;

            while (pq.count() > 0) {
                const current_entry = pq.remove();
                const current = current_entry.vertex;

                // Early exit if we reached the goal
                if (context.eql(current, goal)) {
                    goal_reached = true;
                    break;
                }

                // Skip if already processed
                if (closed.contains(current)) {
                    continue;
                }
                try closed.put(current, {});

                const current_g = g_scores.get(current).?;

                // Get neighbors
                var neighbors_list: std.ArrayList(struct { neighbor: V, weight: W }) = .{};
                defer neighbors_list.deinit(allocator);

                try neighbors_fn(neighbors_ctx, current, &neighbors_list);

                for (neighbors_list.items) |edge| {
                    const neighbor = edge.neighbor;
                    const weight = edge.weight;

                    // Skip if already in closed set
                    if (closed.contains(neighbor)) {
                        continue;
                    }

                    const tentative_g = current_g + weight;

                    // If this path to neighbor is better than any previous one
                    const neighbor_g = g_scores.get(neighbor);
                    if (neighbor_g == null or tentative_g < neighbor_g.?) {
                        try g_scores.put(neighbor, tentative_g);
                        try parents.put(neighbor, current);

                        const h = heuristic_fn(heuristic_ctx, neighbor, goal);
                        const f = tentative_g + h;

                        try pq.add(.{ .vertex = neighbor, .f_score = f });
                    }
                }
            }

            return Result{
                .distances = g_scores,
                .parents = parents,
                .goal_reached = goal_reached,
                .allocator = allocator,
                .context = context,
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const Coord = struct {
    x: i32,
    y: i32,

    pub fn eql(self: Coord, other: Coord) bool {
        return self.x == other.x and self.y == other.y;
    }
};

const CoordContext = struct {
    pub fn hash(_: @This(), key: Coord) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.x));
        hasher.update(std.mem.asBytes(&key.y));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: Coord, b: Coord) bool {
        return a.eql(b);
    }
};

// Test grid with 4-directional movement and obstacle support
const TestGrid = struct {
    width: i32,
    height: i32,
    obstacles: std.AutoHashMap(Coord, void),
    allocator: Allocator,

    fn init(allocator: Allocator, width: i32, height: i32) TestGrid {
        return .{
            .width = width,
            .height = height,
            .obstacles = std.AutoHashMap(Coord, void).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *TestGrid) void {
        self.obstacles.deinit();
    }

    fn addObstacle(self: *TestGrid, x: i32, y: i32) !void {
        try self.obstacles.put(.{ .x = x, .y = y }, {});
    }

    fn isWalkable(self: *const TestGrid, x: i32, y: i32) bool {
        if (x < 0 or x >= self.width or y < 0 or y >= self.height) return false;
        return !self.obstacles.contains(.{ .x = x, .y = y });
    }

    // 4-directional movement (no diagonals)
    fn getNeighbors(self: *const TestGrid, current: Coord, out: anytype) !void {
        const directions = [_][2]i32{
            .{ 0, 1 },   // down
            .{ 0, -1 },  // up
            .{ 1, 0 },   // right
            .{ -1, 0 },  // left
        };

        for (directions) |dir| {
            const nx = current.x + dir[0];
            const ny = current.y + dir[1];
            if (self.isWalkable(nx, ny)) {
                try out.append(self.allocator, .{ .neighbor = .{ .x = nx, .y = ny }, .weight = 1 });
            }
        }
    }
};

// Manhattan distance heuristic
fn manhattanHeuristic(_: *const TestGrid, current: Coord, goal: Coord) i32 {
    return @intCast(@as(i32, @intCast(@abs(goal.x - current.x))) + @as(i32, @intCast(@abs(goal.y - current.y))));
}

// Euclidean distance heuristic (admissible for grid)
fn euclideanHeuristic(_: *const TestGrid, current: Coord, goal: Coord) i32 {
    const dx = goal.x - current.x;
    const dy = goal.y - current.y;
    const dist_sq = dx * dx + dy * dy;
    return @intFromFloat(@sqrt(@as(f32, @floatFromInt(dist_sq))));
}

// Zero heuristic (equivalent to Dijkstra)
fn zeroHeuristic(ctx: *const TestGrid, current: Coord, goal: Coord) i32 {
    _ = ctx;
    _ = current;
    _ = goal;
    return 0;
}

// Non-admissible heuristic (overestimates)
fn nonAdmissibleHeuristic(_: *const TestGrid, current: Coord, goal: Coord) i32 {
    const dx = @abs(goal.x - current.x);
    const dy = @abs(goal.y - current.y);
    return @intCast(dx + dy + 5); // Overestimates by 5
}

test "A*: simple path on open grid" {
    var grid = TestGrid.init(testing.allocator, 5, 5);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 4, .y = 4 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    // Should find goal on open grid
    try testing.expect(result.goal_reached);

    // Check path exists
    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);

    // Path length should be optimal (8 steps for Manhattan distance 8)
    try testing.expectEqual(@as(usize, 9), path.len); // 9 nodes, 8 steps

    // Path should start at start and end at goal
    try testing.expect(path[0].eql(start));
    try testing.expect(path[path.len - 1].eql(goal));

    // Check distance
    try testing.expectEqual(@as(i32, 8), result.getDistance(goal).?);
}

test "A*: grid with obstacles requiring detour" {
    var grid = TestGrid.init(testing.allocator, 5, 5);
    defer grid.deinit();

    // Add vertical wall at x=2, except at y=2
    try grid.addObstacle(2, 0);
    try grid.addObstacle(2, 1);
    try grid.addObstacle(2, 3);
    try grid.addObstacle(2, 4);

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 2 };
    const goal = Coord{ .x = 4, .y = 2 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);

    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);

    // Path should go around the wall
    try testing.expectEqual(@as(usize, 5), path.len); // 4 + 1 node
    try testing.expect(path[0].eql(start));
    try testing.expect(path[path.len - 1].eql(goal));
    try testing.expectEqual(@as(i32, 4), result.getDistance(goal).?);
}

test "A*: no path when goal surrounded by obstacles" {
    var grid = TestGrid.init(testing.allocator, 5, 5);
    defer grid.deinit();

    // Surround goal at (2,2) with obstacles
    try grid.addObstacle(1, 2);
    try grid.addObstacle(3, 2);
    try grid.addObstacle(2, 1);
    try grid.addObstacle(2, 3);

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 2, .y = 2 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    // Goal should not be reached
    try testing.expect(!result.goal_reached);

    // Goal distance should not be computed
    try testing.expectEqual(@as(?i32, null), result.getDistance(goal));

    // Path reconstruction should return null
    try testing.expectEqual(@as(?[]Coord, null), try result.getPath(goal));
}

test "A*: start equals goal" {
    var grid = TestGrid.init(testing.allocator, 3, 3);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const pos = Coord{ .x = 1, .y = 1 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        pos,
        pos,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    // Should find goal immediately
    try testing.expect(result.goal_reached);

    // Distance should be 0
    try testing.expectEqual(@as(i32, 0), result.getDistance(pos).?);

    // Path should have single element
    const path = (try result.getPath(pos)).?;
    defer testing.allocator.free(path);
    try testing.expectEqual(@as(usize, 1), path.len);
    try testing.expect(path[0].eql(pos));
}

test "A*: single vertex graph" {
    var grid = TestGrid.init(testing.allocator, 1, 1);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const pos = Coord{ .x = 0, .y = 0 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        pos,
        pos,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);
    try testing.expectEqual(@as(i32, 0), result.getDistance(pos).?);
}

test "A*: isolated vertex (no neighbors)" {
    var grid = TestGrid.init(testing.allocator, 1, 1);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 0, .y = 0 }; // Same as start (only way to reach)

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);
}

test "A*: multiple paths - heuristic guides to optimal" {
    var grid = TestGrid.init(testing.allocator, 5, 3);
    defer grid.deinit();

    // Two routes from (0,1) to (4,1):
    // Route 1: Direct (4 steps)
    // Route 2: Up and around (6 steps)

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 1 };
    const goal = Coord{ .x = 4, .y = 1 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);

    // Should choose optimal path (direct = 4 steps, 5 nodes)
    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);

    try testing.expectEqual(@as(i32, 4), result.getDistance(goal).?);
    try testing.expectEqual(@as(usize, 5), path.len);
}

test "A*: Manhattan heuristic" {
    var grid = TestGrid.init(testing.allocator, 6, 6);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 5, .y = 5 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);
    try testing.expectEqual(@as(i32, 10), result.getDistance(goal).?);
}

test "A*: Euclidean heuristic" {
    var grid = TestGrid.init(testing.allocator, 5, 5);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 3, .y = 4 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        euclideanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);
    // Actual path length is 7 (3 right, 4 down)
    try testing.expectEqual(@as(i32, 7), result.getDistance(goal).?);
}

test "A*: zero heuristic (degrades to Dijkstra)" {
    var grid = TestGrid.init(testing.allocator, 4, 4);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 3, .y = 3 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        zeroHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    // With zero heuristic, should still find path but may explore more
    try testing.expect(result.goal_reached);
    try testing.expectEqual(@as(i32, 6), result.getDistance(goal).?);
}

test "A*: non-admissible heuristic" {
    var grid = TestGrid.init(testing.allocator, 4, 4);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 3, .y = 3 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        nonAdmissibleHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    // Non-admissible heuristic may find goal but distance is still correct
    try testing.expect(result.goal_reached);
    try testing.expectEqual(@as(i32, 6), result.getDistance(goal).?);
}

test "A*: large open grid" {
    var grid = TestGrid.init(testing.allocator, 50, 50);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 49, .y = 49 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);

    // Manhattan distance is 98
    try testing.expectEqual(@as(i32, 98), result.getDistance(goal).?);

    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);

    // Optimal path has 99 nodes
    try testing.expectEqual(@as(usize, 99), path.len);
}

test "A*: complex maze with solution" {
    var grid = TestGrid.init(testing.allocator, 7, 7);
    defer grid.deinit();

    // Create a maze pattern
    try grid.addObstacle(1, 0);
    try grid.addObstacle(1, 1);
    try grid.addObstacle(1, 2);
    try grid.addObstacle(3, 2);
    try grid.addObstacle(3, 3);
    try grid.addObstacle(3, 4);
    try grid.addObstacle(5, 4);
    try grid.addObstacle(5, 5);

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 6, .y = 6 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);

    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);

    // Verify path validity - each step should be adjacent
    for (0..path.len - 1) |i| {
        const curr = path[i];
        const next = path[i + 1];
        const dx = @abs(next.x - curr.x);
        const dy = @abs(next.y - curr.y);
        try testing.expect((dx == 1 and dy == 0) or (dx == 0 and dy == 1));
    }
}

test "A*: adjacent vertices" {
    var grid = TestGrid.init(testing.allocator, 2, 2);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 1, .y = 0 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);
    try testing.expectEqual(@as(i32, 1), result.getDistance(goal).?);

    const path = (try result.getPath(goal)).?;
    defer testing.allocator.free(path);
    try testing.expectEqual(@as(usize, 2), path.len);
}

test "A*: memory cleanup (no leaks)" {
    var grid = TestGrid.init(testing.allocator, 10, 10);
    defer grid.deinit();

    const Algo = AStar(Coord, i32, CoordContext);
    const start = Coord{ .x = 0, .y = 0 };
    const goal = Coord{ .x = 9, .y = 9 };

    var result = try Algo.run(
        testing.allocator,
        CoordContext{},
        start,
        goal,
        TestGrid.getNeighbors,
        manhattanHeuristic,
        &grid,
        &grid,
    );
    defer result.deinit();

    try testing.expect(result.goal_reached);

    // Allocate and free path to check for leaks
    if (try result.getPath(goal)) |path| {
        testing.allocator.free(path);
    }

    // std.testing.allocator will report leaks on test completion
}
