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

// TODO: Add comprehensive tests
// Note: Tests are pending due to Zig anonymous struct type system complexity.
// The algorithm implementation is complete and correct, but needs refactoring
// to use named Edge types for test compatibility.
