/// Approximation Algorithms
///
/// Efficient algorithms for NP-hard optimization problems with provable approximation guarantees.
/// These algorithms provide solutions within a bounded factor of the optimal solution.

pub const vertex_cover = @import("approximation/vertex_cover.zig");
pub const set_cover = @import("approximation/set_cover.zig");
pub const bin_packing = @import("approximation/bin_packing.zig");
pub const tsp = @import("approximation/tsp.zig");

// Re-export main functions for convenience

// Vertex Cover (2-approximation)
pub const vertexCoverApprox = vertex_cover.vertexCoverApprox;
pub const vertexCoverGreedy = vertex_cover.vertexCoverGreedy;
pub const isValidVertexCover = vertex_cover.isValidCover;

// Set Cover (O(log n)-approximation)
pub const setCoverGreedy = set_cover.setCoverGreedy;
pub const setCoverFrequency = set_cover.setCoverFrequency;
pub const isValidSetCover = set_cover.isValidCover;

// Bin Packing (various approximations)
pub const firstFit = bin_packing.firstFit;
pub const bestFit = bin_packing.bestFit;
pub const firstFitDecreasing = bin_packing.firstFitDecreasing;
pub const BinPackingResult = bin_packing.BinPackingResult;
pub const isValidPacking = bin_packing.isValidPacking;
pub const totalItems = bin_packing.totalItems;

// TSP (2-approximation for metric TSP)
pub const tspMst = tsp.tspMst;
pub const tspNearestNeighbor = tsp.tspNearestNeighbor;
pub const TspResult = tsp.TspResult;
pub const isValidTour = tsp.isValidTour;
pub const tourCost = tsp.tourCost;
