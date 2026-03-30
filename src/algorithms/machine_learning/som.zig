const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Self-Organizing Map (SOM) / Kohonen Network
///
/// Self-Organizing Map is an unsupervised learning algorithm that produces
/// a low-dimensional (typically 2D) representation of high-dimensional data.
/// Unlike t-SNE, SOM uses a competitive learning process to organize neurons
/// in a topological grid that preserves neighborhood relationships.
///
/// Algorithm: Competitive Learning with Neighborhood Function
/// 1. Initialize weight vectors for each neuron in the grid
/// 2. For each input vector:
///    a. Find Best Matching Unit (BMU) - neuron with closest weight vector
///    b. Update BMU and its neighbors towards the input vector
///    c. Decrease learning rate and neighborhood radius over time
/// 3. Converge after fixed iterations or when weight changes stabilize
///
/// Time complexity: O(iterations × n × (grid_width × grid_height × d))
/// Space complexity: O((grid_width × grid_height) × d) for weight vectors
///
/// Applications:
/// - Data visualization (high-dimensional data → 2D grid)
/// - Clustering and pattern recognition
/// - Feature extraction and dimensionality reduction
/// - Anomaly detection (outliers map to edge neurons)
/// - Time series analysis (temporal patterns)
///
/// Type parameters:
/// - T: Numeric type (f32 or f64)
pub fn SOM(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Grid dimensions
        width: usize,
        height: usize,

        /// Input dimension
        input_dim: usize,

        /// Weight vectors: [width * height][input_dim]
        weights: [][]T,

        /// Training parameters
        initial_learning_rate: T,
        initial_radius: T,
        iterations: usize,

        /// Allocator for memory management
        allocator: Allocator,

        /// Neighborhood function type
        pub const NeighborhoodFunction = enum {
            gaussian, // Gaussian decay (smooth)
            bubble, // Step function (hard cutoff)
            mexican_hat, // Mexican hat wavelet (excitation + inhibition)
        };

        /// Initialization method
        pub const InitMethod = enum {
            random, // Random weights from data range
            pca, // Linear initialization along principal components
            sample, // Sample from input data
        };

        /// Training configuration
        pub const Config = struct {
            initial_learning_rate: T = 0.5,
            initial_radius: T = 0.0, // 0 = auto (max(width, height) / 2)
            iterations: usize = 1000,
            neighborhood_fn: NeighborhoodFunction = .gaussian,
            init_method: InitMethod = .sample,
        };

        /// Coordinate in the SOM grid
        pub const Coord = struct {
            x: usize,
            y: usize,

            /// Euclidean distance between two grid coordinates
            pub fn distance(self: Coord, other: Coord) T {
                const dx = @as(T, @floatFromInt(self.x)) - @as(T, @floatFromInt(other.x));
                const dy = @as(T, @floatFromInt(self.y)) - @as(T, @floatFromInt(other.y));
                return @sqrt(dx * dx + dy * dy);
            }
        };

        /// Initialize SOM with given dimensions
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(width × height × input_dim)
        pub fn init(
            allocator: Allocator,
            width: usize,
            height: usize,
            input_dim: usize,
            config: Config,
        ) !Self {
            if (width == 0 or height == 0) return error.InvalidGridSize;
            if (input_dim == 0) return error.InvalidInputDimension;

            const num_neurons = width * height;
            const weights = try allocator.alloc([]T, num_neurons);
            errdefer allocator.free(weights);

            for (weights, 0..) |*w, i| {
                w.* = try allocator.alloc(T, input_dim);
                errdefer {
                    for (weights[0..i]) |prev| allocator.free(prev);
                    allocator.free(weights);
                }
            }

            const radius = if (config.initial_radius == 0)
                @as(T, @floatFromInt(@max(width, height))) / 2.0
            else
                config.initial_radius;

            return Self{
                .width = width,
                .height = height,
                .input_dim = input_dim,
                .weights = weights,
                .initial_learning_rate = config.initial_learning_rate,
                .initial_radius = radius,
                .iterations = config.iterations,
                .allocator = allocator,
            };
        }

        /// Free all allocated memory
        ///
        /// Time: O(width × height)
        /// Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.weights) |w| {
                self.allocator.free(w);
            }
            self.allocator.free(self.weights);
        }

        /// Initialize weights from input data
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(1)
        pub fn initWeights(
            self: *Self,
            data: []const []const T,
            method: InitMethod,
            seed: u64,
        ) !void {
            if (data.len == 0) return error.EmptyData;
            if (data[0].len != self.input_dim) return error.DimensionMismatch;

            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();

            switch (method) {
                .random => {
                    // Random weights within data range
                    var min_vals = try self.allocator.alloc(T, self.input_dim);
                    defer self.allocator.free(min_vals);
                    var max_vals = try self.allocator.alloc(T, self.input_dim);
                    defer self.allocator.free(max_vals);

                    // Find min/max for each dimension
                    for (0..self.input_dim) |d| {
                        min_vals[d] = data[0][d];
                        max_vals[d] = data[0][d];
                        for (data[1..]) |point| {
                            if (point[d] < min_vals[d]) min_vals[d] = point[d];
                            if (point[d] > max_vals[d]) max_vals[d] = point[d];
                        }
                    }

                    // Initialize each neuron with random weights
                    for (self.weights) |w| {
                        for (w, 0..) |*val, d| {
                            const range = max_vals[d] - min_vals[d];
                            val.* = min_vals[d] + random.float(T) * range;
                        }
                    }
                },
                .sample => {
                    // Sample random data points
                    for (self.weights) |w| {
                        const idx = random.uintLessThan(usize, data.len);
                        @memcpy(w, data[idx]);
                    }
                },
                .pca => {
                    // Simple linear initialization (approximation of PCA)
                    // Initialize along two principal axes
                    for (self.weights, 0..) |w, i| {
                        const row = i / self.width;
                        const col = i % self.width;
                        const t1 = @as(T, @floatFromInt(col)) / @as(T, @floatFromInt(self.width - 1));
                        const t2 = @as(T, @floatFromInt(row)) / @as(T, @floatFromInt(self.height - 1));

                        // Linear interpolation between extremes
                        for (w, 0..) |*val, d| {
                            var min_val = data[0][d];
                            var max_val = data[0][d];
                            for (data[1..]) |point| {
                                if (point[d] < min_val) min_val = point[d];
                                if (point[d] > max_val) max_val = point[d];
                            }
                            // Interpolate along both axes
                            val.* = min_val + (max_val - min_val) * (t1 + t2) / 2.0;
                        }
                    }
                },
            }
        }

        /// Train the SOM on input data
        ///
        /// Time: O(iterations × n × (width × height × d))
        /// Space: O(1)
        pub fn train(
            self: *Self,
            data: []const []const T,
            neighborhood_fn: NeighborhoodFunction,
        ) !void {
            if (data.len == 0) return error.EmptyData;
            if (data[0].len != self.input_dim) return error.DimensionMismatch;

            var iteration: usize = 0;
            while (iteration < self.iterations) : (iteration += 1) {
                // Decay parameters
                const progress = @as(T, @floatFromInt(iteration)) / @as(T, @floatFromInt(self.iterations));
                const learning_rate = self.initial_learning_rate * (1.0 - progress);
                const radius = self.initial_radius * (1.0 - progress);

                // Process each data point
                for (data) |point| {
                    // Find Best Matching Unit (BMU)
                    const bmu = try self.findBMU(point);

                    // Update BMU and neighbors
                    try self.updateWeights(point, bmu, learning_rate, radius, neighborhood_fn);
                }
            }
        }

        /// Find Best Matching Unit (neuron with closest weight vector)
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(1)
        pub fn findBMU(self: *Self, input: []const T) !Coord {
            if (input.len != self.input_dim) return error.DimensionMismatch;

            var best_coord = Coord{ .x = 0, .y = 0 };
            var best_dist = std.math.inf(T);

            for (self.weights, 0..) |w, i| {
                const dist = euclideanDistance(T, input, w);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_coord = .{
                        .x = i % self.width,
                        .y = i / self.width,
                    };
                }
            }

            return best_coord;
        }

        /// Update weights of BMU and its neighbors
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(1)
        fn updateWeights(
            self: *Self,
            input: []const T,
            bmu: Coord,
            learning_rate: T,
            radius: T,
            neighborhood_fn: NeighborhoodFunction,
        ) !void {
            for (self.weights, 0..) |w, i| {
                const coord = Coord{
                    .x = i % self.width,
                    .y = i / self.width,
                };

                const dist = bmu.distance(coord);
                const influence = neighborhoodInfluence(T, dist, radius, neighborhood_fn);

                // Update weight: w = w + influence × learning_rate × (input - w)
                for (w, 0..) |*weight, d| {
                    const delta = input[d] - weight.*;
                    weight.* += influence * learning_rate * delta;
                }
            }
        }

        /// Map input vector to grid coordinates (find BMU)
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(1)
        pub fn map(self: *Self, input: []const T) !Coord {
            return self.findBMU(input);
        }

        /// Get weight vector for a grid position
        ///
        /// Time: O(1)
        /// Space: O(1)
        pub fn getWeights(self: *Self, coord: Coord) ![]const T {
            if (coord.x >= self.width or coord.y >= self.height) {
                return error.OutOfBounds;
            }
            const idx = coord.y * self.width + coord.x;
            return self.weights[idx];
        }

        /// Compute U-Matrix (unified distance matrix) for visualization
        /// Shows average distance to neighbors - useful for cluster boundary detection
        ///
        /// Time: O(width × height × input_dim)
        /// Space: O(width × height)
        pub fn computeUMatrix(self: *Self, allocator: Allocator) ![]T {
            const umatrix = try allocator.alloc(T, self.width * self.height);
            errdefer allocator.free(umatrix);

            for (0..self.height) |y| {
                for (0..self.width) |x| {
                    const idx = y * self.width + x;
                    const weights = self.weights[idx];

                    // Average distance to neighbors
                    var sum: T = 0;
                    var count: usize = 0;

                    // Check 4-neighborhood (up, down, left, right)
                    const neighbors = [_]struct { dx: isize, dy: isize }{
                        .{ .dx = -1, .dy = 0 },
                        .{ .dx = 1, .dy = 0 },
                        .{ .dx = 0, .dy = -1 },
                        .{ .dx = 0, .dy = 1 },
                    };

                    for (neighbors) |n| {
                        const nx = @as(isize, @intCast(x)) + n.dx;
                        const ny = @as(isize, @intCast(y)) + n.dy;
                        if (nx >= 0 and nx < @as(isize, @intCast(self.width)) and
                            ny >= 0 and ny < @as(isize, @intCast(self.height)))
                        {
                            const nidx = @as(usize, @intCast(ny)) * self.width + @as(usize, @intCast(nx));
                            sum += euclideanDistance(T, weights, self.weights[nidx]);
                            count += 1;
                        }
                    }

                    umatrix[idx] = if (count > 0) sum / @as(T, @floatFromInt(count)) else 0;
                }
            }

            return umatrix;
        }

        /// Compute quantization error (average distance from data points to their BMUs)
        ///
        /// Time: O(n × width × height × input_dim)
        /// Space: O(1)
        pub fn quantizationError(self: *Self, data: []const []const T) !T {
            if (data.len == 0) return error.EmptyData;

            var total_error: T = 0;
            for (data) |point| {
                const bmu = try self.findBMU(point);
                const bmu_weights = try self.getWeights(bmu);
                total_error += euclideanDistance(T, point, bmu_weights);
            }

            return total_error / @as(T, @floatFromInt(data.len));
        }

        /// Compute topographic error (percentage of data points where BMU and 2nd BMU are not adjacent)
        ///
        /// Time: O(n × width × height × input_dim)
        /// Space: O(1)
        pub fn topographicError(self: *Self, data: []const []const T) !T {
            if (data.len == 0) return error.EmptyData;

            var non_adjacent_count: usize = 0;
            for (data) |point| {
                // Find BMU and 2nd BMU
                var best_dist = std.math.inf(T);
                var second_best_dist = std.math.inf(T);
                var bmu: Coord = undefined;
                var second_bmu: Coord = undefined;

                for (self.weights, 0..) |w, i| {
                    const dist = euclideanDistance(T, point, w);
                    const coord = Coord{
                        .x = i % self.width,
                        .y = i / self.width,
                    };

                    if (dist < best_dist) {
                        second_best_dist = best_dist;
                        second_bmu = bmu;
                        best_dist = dist;
                        bmu = coord;
                    } else if (dist < second_best_dist) {
                        second_best_dist = dist;
                        second_bmu = coord;
                    }
                }

                // Check if BMU and 2nd BMU are adjacent
                const grid_dist = bmu.distance(second_bmu);
                if (grid_dist > 1.5) { // Not adjacent (allows diagonal)
                    non_adjacent_count += 1;
                }
            }

            return @as(T, @floatFromInt(non_adjacent_count)) / @as(T, @floatFromInt(data.len));
        }
    };
}

/// Euclidean distance between two vectors
///
/// Time: O(n)
/// Space: O(1)
fn euclideanDistance(comptime T: type, a: []const T, b: []const T) T {
    var sum: T = 0;
    for (a, b) |av, bv| {
        const diff = av - bv;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Neighborhood influence function
///
/// Time: O(1)
/// Space: O(1)
fn neighborhoodInfluence(
    comptime T: type,
    distance: T,
    radius: T,
    function: SOM(T).NeighborhoodFunction,
) T {
    return switch (function) {
        .gaussian => blk: {
            // Gaussian: exp(-distance² / (2 × radius²))
            const d2 = distance * distance;
            const r2 = radius * radius;
            if (r2 == 0) break :blk if (distance == 0) 1.0 else 0.0;
            break :blk @exp(-d2 / (2.0 * r2));
        },
        .bubble => blk: {
            // Bubble: 1 if distance ≤ radius, else 0
            break :blk if (distance <= radius) 1.0 else 0.0;
        },
        .mexican_hat => blk: {
            // Mexican hat: (1 - distance²/radius²) × exp(-distance²/(2×radius²))
            const d2 = distance * distance;
            const r2 = radius * radius;
            if (r2 == 0) break :blk if (distance == 0) 1.0 else 0.0;
            const gauss = @exp(-d2 / (2.0 * r2));
            break :blk (1.0 - d2 / r2) * gauss;
        },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "SOM: initialization" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 5, 5, 3, .{});
    defer som.deinit();

    try testing.expectEqual(@as(usize, 5), som.width);
    try testing.expectEqual(@as(usize, 5), som.height);
    try testing.expectEqual(@as(usize, 3), som.input_dim);
    try testing.expectEqual(@as(usize, 25), som.weights.len);
}

test "SOM: invalid parameters" {
    const allocator = testing.allocator;

    try testing.expectError(error.InvalidGridSize, SOM(f32).init(allocator, 0, 5, 3, .{}));
    try testing.expectError(error.InvalidGridSize, SOM(f32).init(allocator, 5, 0, 3, .{}));
    try testing.expectError(error.InvalidInputDimension, SOM(f32).init(allocator, 5, 5, 0, .{}));
}

test "SOM: weight initialization - random" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
        &[_]f32{ 2.0, 2.0 },
    };

    try som.initWeights(&data, .random, 42);

    // Weights should be within data range
    for (som.weights) |w| {
        for (w) |val| {
            try testing.expect(val >= 0.0 and val <= 2.0);
        }
    }
}

test "SOM: weight initialization - sample" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
        &[_]f32{ 2.0, 2.0 },
    };

    try som.initWeights(&data, .sample, 42);

    // Each weight should be a copy of some data point
    for (som.weights) |w| {
        var found = false;
        for (data) |point| {
            if (w[0] == point[0] and w[1] == point[1]) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "SOM: find BMU" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    // Manually set weights for predictable BMU
    for (som.weights, 0..) |w, i| {
        w[0] = @floatFromInt(i);
        w[1] = @floatFromInt(i);
    }

    const input = [_]f32{ 4.1, 4.1 };
    const bmu = try som.findBMU(&input);

    // BMU should be neuron 4 (weights [4, 4])
    try testing.expectEqual(@as(usize, 1), bmu.x);
    try testing.expectEqual(@as(usize, 1), bmu.y);
}

test "SOM: training - gaussian neighborhood" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 5, 5, 2, .{
        .iterations = 100,
        .initial_learning_rate = 0.5,
    });
    defer som.deinit();

    // Two clusters
    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 }, &[_]f32{ 0.1, 0.0 }, &[_]f32{ 0.0, 0.1 },
        &[_]f32{ 1.0, 1.0 }, &[_]f32{ 1.1, 1.0 }, &[_]f32{ 1.0, 1.1 },
    };

    try som.initWeights(&data, .random, 42);
    try som.train(&data, .gaussian);

    // After training, nearby neurons should have similar weights
    const coord1 = SOM(f32).Coord{ .x = 2, .y = 2 };
    const coord2 = SOM(f32).Coord{ .x = 2, .y = 3 };
    const w1 = try som.getWeights(coord1);
    const w2 = try som.getWeights(coord2);

    const dist = euclideanDistance(f32, w1, w2);
    try testing.expect(dist < 0.5); // Neighbors should be similar
}

test "SOM: training - bubble neighborhood" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 4, 4, 2, .{
        .iterations = 50,
    });
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .sample, 42);
    try som.train(&data, .bubble);

    // Should converge
    const qe = try som.quantizationError(&data);
    try testing.expect(qe < 1.0);
}

test "SOM: mapping" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 4, 4, 2, .{ .iterations = 100 });
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .sample, 42);
    try som.train(&data, .gaussian);

    // Map should find reasonable coordinates
    const input = [_]f32{ 0.5, 0.5 };
    const coord = try som.map(&input);

    try testing.expect(coord.x < som.width);
    try testing.expect(coord.y < som.height);
}

test "SOM: U-Matrix" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .sample, 42);

    const umatrix = try som.computeUMatrix(allocator);
    defer allocator.free(umatrix);

    try testing.expectEqual(@as(usize, 9), umatrix.len);
    // All values should be non-negative
    for (umatrix) |val| {
        try testing.expect(val >= 0.0);
    }
}

test "SOM: quantization error" {
    const allocator = testing.allocator;

    var som = try SOM(f64).init(allocator, 5, 5, 2, .{ .iterations = 100 });
    defer som.deinit();

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 }, &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 1.0, 1.0 }, &[_]f64{ 1.1, 1.1 },
    };

    try som.initWeights(&data, .sample, 42);

    const error_before = try som.quantizationError(&data);
    try som.train(&data, .gaussian);
    const error_after = try som.quantizationError(&data);

    // Error should decrease after training
    try testing.expect(error_after < error_before);
    try testing.expect(error_after >= 0.0);
}

test "SOM: topographic error" {
    const allocator = testing.allocator;

    var som = try SOM(f64).init(allocator, 5, 5, 2, .{ .iterations = 100 });
    defer som.deinit();

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .sample, 42);
    try som.train(&data, .gaussian);

    const topo_error = try som.topographicError(&data);

    // Topographic error should be between 0 and 1
    try testing.expect(topo_error >= 0.0);
    try testing.expect(topo_error <= 1.0);
}

test "SOM: neighborhood functions" {
    // Gaussian
    {
        const influence = neighborhoodInfluence(f32, 0.0, 1.0, .gaussian);
        try testing.expectApproxEqAbs(@as(f32, 1.0), influence, 1e-6);

        const influence2 = neighborhoodInfluence(f32, 2.0, 1.0, .gaussian);
        try testing.expect(influence2 < 0.5);
    }

    // Bubble
    {
        const influence1 = neighborhoodInfluence(f32, 0.5, 1.0, .bubble);
        try testing.expectEqual(@as(f32, 1.0), influence1);

        const influence2 = neighborhoodInfluence(f32, 1.5, 1.0, .bubble);
        try testing.expectEqual(@as(f32, 0.0), influence2);
    }

    // Mexican hat
    {
        const influence = neighborhoodInfluence(f32, 0.0, 1.0, .mexican_hat);
        try testing.expectApproxEqAbs(@as(f32, 1.0), influence, 1e-6);
    }
}

test "SOM: grid coordinate distance" {
    const coord1 = SOM(f32).Coord{ .x = 0, .y = 0 };
    const coord2 = SOM(f32).Coord{ .x = 3, .y = 4 };

    const dist = coord1.distance(coord2);
    try testing.expectApproxEqAbs(@as(f32, 5.0), dist, 1e-6); // 3-4-5 triangle
}

test "SOM: f32 type support" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{ .iterations = 50 });
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .random, 42);
    try som.train(&data, .gaussian);

    const qe = try som.quantizationError(&data);
    try testing.expect(qe >= 0.0);
}

test "SOM: large grid" {
    const allocator = testing.allocator;

    var som = try SOM(f64).init(allocator, 10, 10, 3, .{ .iterations = 50 });
    defer som.deinit();

    const data = [_][]const f64{
        &[_]f64{ 0.0, 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0, 1.0 },
        &[_]f64{ 2.0, 2.0, 2.0 },
    };

    try som.initWeights(&data, .sample, 42);
    try som.train(&data, .gaussian);

    try testing.expectEqual(@as(usize, 100), som.weights.len);
}

test "SOM: mexican hat neighborhood" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 4, 4, 2, .{ .iterations = 50 });
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
    };

    try som.initWeights(&data, .sample, 42);
    try som.train(&data, .mexican_hat);

    const qe = try som.quantizationError(&data);
    try testing.expect(qe >= 0.0);
}

test "SOM: dimension mismatch" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const wrong_data = [_][]const f32{
        &[_]f32{ 0.0, 0.0, 0.0 }, // 3D instead of 2D
    };

    try testing.expectError(error.DimensionMismatch, som.initWeights(&wrong_data, .sample, 42));
}

test "SOM: empty data" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const empty_data: []const []const f32 = &[_][]const f32{};

    try testing.expectError(error.EmptyData, som.initWeights(empty_data, .sample, 42));
    try testing.expectError(error.EmptyData, som.train(empty_data, .gaussian));
}

test "SOM: get weights out of bounds" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    try testing.expectError(error.OutOfBounds, som.getWeights(.{ .x = 5, .y = 2 }));
    try testing.expectError(error.OutOfBounds, som.getWeights(.{ .x = 2, .y = 5 }));
}

test "SOM: pca initialization" {
    const allocator = testing.allocator;

    var som = try SOM(f32).init(allocator, 3, 3, 2, .{});
    defer som.deinit();

    const data = [_][]const f32{
        &[_]f32{ 0.0, 0.0 },
        &[_]f32{ 1.0, 1.0 },
        &[_]f32{ 2.0, 2.0 },
    };

    try som.initWeights(&data, .pca, 42);

    // Weights should be linearly distributed
    for (som.weights) |w| {
        for (w) |val| {
            try testing.expect(val >= 0.0 and val <= 2.0);
        }
    }
}
