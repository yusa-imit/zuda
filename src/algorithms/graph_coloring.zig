// ============================================================================
// Graph Coloring Algorithms
// ============================================================================
//
// This module provides comprehensive graph coloring algorithms for vertex coloring problems.
// Graph coloring is the assignment of colors to vertices such that no two adjacent vertices
// share the same color. The chromatic number is the minimum number of colors needed.
//
// ## Algorithms Included
//
// ### Greedy Heuristics (greedy_coloring.zig)
// - **greedyColoring**: O(V²) simple first-fit strategy
// - **welshPowell**: O(V² + E) degree-based ordering (often better results)
// - **dsatur**: O(V² + E) saturation-based ordering (near-optimal results)
// - **chromaticNumber**: O(V) compute number of colors used
// - **isValidColoring**: O(V + E) validate a coloring
//
// ### Exact Algorithms (backtracking_coloring.zig)
// - **mColoring**: O(m^V) determine if graph is m-colorable
// - **findChromaticNumber**: O(V^(V+1)) find minimum colors (small graphs only)
// - **exactKColoring**: O(m^V) check if exactly k colors are needed
// - **coloringWithPropagation**: O(m^V) backtracking with constraint propagation
//
// ## Applications
//
// - **Register Allocation**: Assigning CPU registers to program variables
// - **Scheduling**: Time slot assignment with conflict constraints
// - **Map Coloring**: Geographical region coloring
// - **Frequency Assignment**: Radio frequency allocation to minimize interference
// - **Sudoku Solving**: Each row/column/block is a clique
// - **Pattern Matching**: Finding independent sets
//
// ## Algorithm Selection Guide
//
// | Problem Type | Algorithm | Complexity | Quality |
// |--------------|-----------|------------|---------|
// | Fast approximation | greedyColoring | O(V²) | Good |
// | Better approximation | welshPowell | O(V² + E) | Better |
// | Best approximation | dsatur | O(V² + E) | Best |
// | Exact solution (small) | mColoring | O(m^V) | Optimal |
// | Find chromatic number | findChromaticNumber | O(V^(V+1)) | Optimal |
// | With pruning | coloringWithPropagation | O(m^V) | Optimal |
//
// ## Complexity Notes
//
// - **Graph coloring is NP-complete** — exact algorithms have exponential worst-case complexity
// - For large graphs (V > 20), use greedy heuristics (DSatur often gives near-optimal results)
// - For small graphs (V < 15), exact algorithms are feasible
// - Chromatic number bounds: χ(G) ≤ Δ(G) + 1 where Δ is maximum degree (Brook's theorem)
//
// ## Usage Examples
//
// ### Greedy Coloring (Fast)
// ```zig
// const std = @import("std");
// const zuda = @import("zuda");
//
// // Build adjacency list
// var graph = try std.ArrayList(std.ArrayList(u32)).initCapacity(allocator, n);
// // Add edges...
//
// // Color the graph
// const coloring = try zuda.algorithms.graph_coloring.greedyColoring(u32, allocator, graph);
// defer coloring.deinit();
//
// const num_colors = zuda.algorithms.graph_coloring.chromaticNumber(coloring);
// std.debug.print("Used {} colors\n", .{num_colors});
// ```
//
// ### DSatur (Better Quality)
// ```zig
// const coloring = try zuda.algorithms.graph_coloring.dsatur(u32, allocator, graph);
// defer coloring.deinit();
// ```
//
// ### Exact M-Coloring (Small Graphs)
// ```zig
// const result = try zuda.algorithms.graph_coloring.mColoring(u32, allocator, graph, 3);
// if (result) |coloring| {
//     defer coloring.deinit();
//     std.debug.print("Graph is 3-colorable!\n", .{});
// } else {
//     std.debug.print("Graph requires more than 3 colors\n", .{});
// }
// ```
//
// ## Performance Characteristics
//
// ### Greedy Algorithms
// - **greedyColoring**: Simple, fast, uses at most Δ(G) + 1 colors
// - **welshPowell**: Sorts by degree, typically 10-20% better than greedy
// - **dsatur**: Best heuristic, often finds optimal or near-optimal colorings
//
// ### Exact Algorithms
// - Use only for small graphs (V < 15) due to exponential complexity
// - **mColoring**: Determines m-colorability, faster than finding chromatic number
// - **coloringWithPropagation**: Forward checking reduces search space significantly
//
// ## Graph Representation
//
// All functions expect an adjacency list: `ArrayList(ArrayList(T))` where:
// - Index i represents vertex i
// - `graph.items[i]` contains the list of neighbors of vertex i
// - Edges should be bidirectional (if u in neighbors of v, then v in neighbors of u)
//
// ============================================================================

pub const greedy_coloring = @import("graph_coloring/greedy_coloring.zig");
pub const backtracking_coloring = @import("graph_coloring/backtracking_coloring.zig");

// Re-export commonly used functions
pub const greedyColoring = greedy_coloring.greedyColoring;
pub const welshPowell = greedy_coloring.welshPowell;
pub const dsatur = greedy_coloring.dsatur;
pub const chromaticNumber = greedy_coloring.chromaticNumber;
pub const isValidColoring = greedy_coloring.isValidColoring;

pub const mColoring = backtracking_coloring.mColoring;
pub const findChromaticNumber = backtracking_coloring.findChromaticNumber;
pub const exactKColoring = backtracking_coloring.exactKColoring;
pub const coloringWithPropagation = backtracking_coloring.coloringWithPropagation;
