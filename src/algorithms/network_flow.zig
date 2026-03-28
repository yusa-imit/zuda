/// Network Flow Algorithms
///
/// Algorithms for computing maximum flow, minimum cut, and related problems
/// in flow networks. Flow networks model scenarios where resources flow through
/// a network with capacity constraints (e.g., traffic, data transfer, supply chains).
///
/// ## Included Algorithms
///
/// ### Ford-Fulkerson (DFS-based)
/// - **Complexity**: O(E × max_flow)
/// - **Method**: DFS to find augmenting paths
/// - **Use case**: Simple networks, educational purposes
/// - **Advantage**: Easy to understand and implement
/// - **Disadvantage**: Potentially exponential time for irrational capacities
///
/// ### Edmonds-Karp (BFS-based)
/// - **Complexity**: O(V × E²)
/// - **Method**: BFS to find shortest augmenting paths
/// - **Use case**: Guaranteed polynomial time, medium-sized networks
/// - **Advantage**: Polynomial time guarantee, finds shortest paths
/// - **Disadvantage**: Slower than Dinic for large graphs
///
/// ### Dinic's Algorithm (Level Graph)
/// - **Complexity**: O(V² × E) general, O(E × √V) for unit capacity
/// - **Method**: Level graph construction + blocking flow
/// - **Use case**: Dense networks, bipartite matching, large-scale problems
/// - **Advantage**: Fastest for many practical cases, optimal for bipartite matching
/// - **Disadvantage**: More complex implementation
///
/// ## Common Applications
///
/// 1. **Maximum Bipartite Matching**: Find maximum matching in bipartite graph
///    - Job assignment, resource allocation, stable marriages
///    - Use Dinic's algorithm for O(E × √V) time
///
/// 2. **Minimum Cut**: Find minimum capacity edge set that separates source from sink
///    - Network reliability, image segmentation, clustering
///    - Max flow = Min cut (by max-flow min-cut theorem)
///
/// 3. **Circulation with Demands**: Satisfy supply/demand constraints in network
///    - Supply chain optimization, traffic flow, fluid dynamics
///
/// 4. **Multi-commodity Flow**: Route multiple commodities through network
///    - Network routing, bandwidth allocation
///
/// ## Algorithm Selection Guide
///
/// - **Small graphs (<100 vertices)**: Ford-Fulkerson (simple, sufficient)
/// - **Need polynomial guarantee**: Edmonds-Karp
/// - **Large graphs or unit capacity**: Dinic's algorithm
/// - **Bipartite matching**: Dinic's algorithm (optimal O(E√V))
/// - **Min-cut computation**: Any algorithm + min-cut extraction
///
/// ## Example: Maximum Flow
///
/// ```zig
/// const std = @import("std");
/// const zuda = @import("zuda");
/// const network_flow = zuda.algorithms.network_flow;
///
/// pub fn main() !void {
///     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
///     defer _ = gpa.deinit();
///     const allocator = gpa.allocator();
///
///     // Network: s -> 1 -> t
///     //          s -> 2 -> t
///     var capacity = [_][4]u32{
///         .{ 0, 10, 5, 0 },  // s (0)
///         .{ 0, 0, 0, 10 },  // 1
///         .{ 0, 0, 0, 5 },   // 2
///         .{ 0, 0, 0, 0 },   // t (3)
///     };
///     var capacity_ptrs: [4][]const u32 = undefined;
///     for (&capacity, 0..) |*row, i| capacity_ptrs[i] = row;
///
///     // Compute maximum flow using Dinic's algorithm
///     const max_flow = try network_flow.dinic.maxFlow(u32, allocator, &capacity_ptrs, 0, 3);
///     std.debug.print("Maximum flow: {}\n", .{max_flow}); // 15
///
///     // Get minimum cut
///     const min_cut = try network_flow.ford_fulkerson.minCut(u32, allocator, &capacity_ptrs, 0, 3);
///     defer allocator.free(min_cut);
///     std.debug.print("Source side of min-cut: {any}\n", .{min_cut});
/// }
/// ```
///
/// ## Example: Bipartite Matching
///
/// ```zig
/// // Jobs to workers assignment
/// var edges = [_][]const usize{
///     &[_]usize{0, 1},  // Worker 0 can do jobs 0, 1
///     &[_]usize{1, 2},  // Worker 1 can do jobs 1, 2
///     &[_]usize{2},     // Worker 2 can do job 2
/// };
///
/// const matching = try network_flow.dinic.maxBipartiteMatching(
///     allocator, 3, 3, &edges
/// );
/// std.debug.print("Maximum matching: {}\n", .{matching}); // 3 (perfect matching)
/// ```

pub const ford_fulkerson = @import("network_flow/ford_fulkerson.zig");
pub const edmonds_karp = @import("network_flow/edmonds_karp.zig");
pub const dinic = @import("network_flow/dinic.zig");
