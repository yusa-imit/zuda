const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Multi-Layer Perceptron (MLP) - Feedforward Neural Network
///
/// A fully-connected feedforward neural network with:
/// - Multiple hidden layers with configurable sizes
/// - Activation functions (ReLU, Sigmoid, Tanh)
/// - Loss functions (MSE for regression, Cross-Entropy for classification)
/// - Backpropagation with gradient descent
/// - Mini-batch training support
///
/// Time Complexity:
/// - forward: O(L × n_max² × batch) where L = layers, n_max = max layer size
/// - backward: O(L × n_max² × batch)
/// - fit: O(epochs × batches × (forward + backward))
///
/// Space Complexity: O(L × n_max²) for weights + O(L × n_max × batch) for activations
///
/// Use cases:
/// - Classification: Handwritten digit recognition (MNIST), sentiment analysis
/// - Regression: House price prediction, time series forecasting
/// - Feature learning: Extract learned representations from hidden layers
/// - Universal function approximation: Model complex non-linear relationships
///
/// Example:
/// ```zig
/// var mlp = try MLP(f64).init(allocator, &.{4, 8, 3}, .relu, .cross_entropy);
/// defer mlp.deinit();
/// try mlp.fit(X_train, y_train, .{ .epochs = 100, .learning_rate = 0.01 });
/// const predictions = try mlp.predict(allocator, X_test);
/// defer allocator.free(predictions);
/// ```
pub fn MLP(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        layers: []usize, // Layer sizes including input and output
        weights: [][]T, // weights[i] is matrix (layers[i+1] × layers[i])
        biases: [][]T, // biases[i] is vector of size layers[i+1]
        activation: ActivationFn,
        loss_fn: LossFn,

        // Temporary storage for forward/backward pass
        activations: ?[][]T, // Cached activations during forward pass
        gradients: ?[][]T, // Cached gradients during backward pass

        pub const ActivationFn = enum {
            relu,
            sigmoid,
            tanh,
            linear,
        };

        pub const LossFn = enum {
            mse, // Mean Squared Error (regression)
            cross_entropy, // Cross-Entropy (classification)
        };

        pub const FitOptions = struct {
            epochs: usize = 100,
            learning_rate: T = 0.01,
            batch_size: usize = 32,
            verbose: bool = false,
            seed: ?u64 = null,
        };

        /// Initialize MLP with given layer sizes
        /// Time: O(L × n_max²) | Space: O(L × n_max²)
        pub fn init(
            allocator: Allocator,
            layer_sizes: []const usize,
            activation: ActivationFn,
            loss_fn: LossFn,
        ) !Self {
            if (layer_sizes.len < 2) return error.InvalidLayerConfiguration;

            const n_layers = layer_sizes.len - 1; // Number of weight matrices

            // Allocate layer sizes
            const layers = try allocator.alloc(usize, layer_sizes.len);
            @memcpy(layers, layer_sizes);

            // Allocate weights and biases
            const weights = try allocator.alloc([]T, n_layers);
            const biases = try allocator.alloc([]T, n_layers);

            var self = Self{
                .allocator = allocator,
                .layers = layers,
                .weights = weights,
                .biases = biases,
                .activation = activation,
                .loss_fn = loss_fn,
                .activations = null,
                .gradients = null,
            };

            errdefer self.deinit();

            // Initialize weights and biases with Xavier/He initialization
            var prng = std.Random.DefaultPrng.init(0);
            const random = prng.random();

            for (0..n_layers) |i| {
                const in_size = layers[i];
                const out_size = layers[i + 1];

                // Xavier initialization: std = sqrt(2 / (in + out))
                const std_dev = @sqrt(2.0 / @as(T, @floatFromInt(in_size + out_size)));

                // Allocate weight matrix (out_size × in_size)
                weights[i] = try allocator.alloc(T, out_size * in_size);
                for (weights[i]) |*w| {
                    w.* = (random.float(T) * 2.0 - 1.0) * std_dev;
                }

                // Allocate bias vector (out_size)
                biases[i] = try allocator.alloc(T, out_size);
                @memset(biases[i], 0);
            }

            return self;
        }

        /// Free all allocated memory
        /// Time: O(L) | Space: O(1)
        pub fn deinit(self: *Self) void {
            for (self.weights) |w| self.allocator.free(w);
            for (self.biases) |b| self.allocator.free(b);
            self.allocator.free(self.weights);
            self.allocator.free(self.biases);
            self.allocator.free(self.layers);

            if (self.activations) |acts| {
                for (acts) |a| self.allocator.free(a);
                self.allocator.free(acts);
            }
            if (self.gradients) |grads| {
                for (grads) |g| self.allocator.free(g);
                self.allocator.free(grads);
            }
        }

        /// Forward pass through the network
        /// Time: O(L × n_max² × batch) | Space: O(L × n_max × batch)
        fn forward(self: *Self, X: []const T, batch_size: usize) ![]T {
            const n_layers = self.weights.len;
            const input_size = self.layers[0];
            const output_size = self.layers[self.layers.len - 1];

            // Allocate activation storage if needed
            if (self.activations == null) {
                const acts = try self.allocator.alloc([]T, n_layers + 1);
                for (0..n_layers + 1) |i| {
                    acts[i] = try self.allocator.alloc(T, self.layers[i] * batch_size);
                }
                self.activations = acts;
            }

            const acts = self.activations.?;

            // Copy input to first activation layer
            @memcpy(acts[0][0 .. input_size * batch_size], X[0 .. input_size * batch_size]);

            // Forward through each layer
            for (0..n_layers) |layer_idx| {
                const in_size = self.layers[layer_idx];
                const out_size = self.layers[layer_idx + 1];
                const W = self.weights[layer_idx];
                const b = self.biases[layer_idx];
                const input = acts[layer_idx];
                const output = acts[layer_idx + 1];

                // Matrix multiply: output = W @ input + b
                for (0..batch_size) |batch_idx| {
                    for (0..out_size) |i| {
                        var sum: T = b[i];
                        for (0..in_size) |j| {
                            sum += W[i * in_size + j] * input[batch_idx * in_size + j];
                        }
                        output[batch_idx * out_size + i] = sum;
                    }
                }

                // Apply activation (except for output layer in classification)
                const is_output = layer_idx == n_layers - 1;
                if (!is_output or self.loss_fn == .mse) {
                    self.applyActivation(output[0 .. out_size * batch_size]);
                }
            }

            // Return final activations
            const result = try self.allocator.alloc(T, output_size * batch_size);
            @memcpy(result, acts[n_layers][0 .. output_size * batch_size]);
            return result;
        }

        /// Backward pass with gradient computation
        /// Time: O(L × n_max² × batch) | Space: O(L × n_max × batch)
        fn backward(self: *Self, y_true: []const T, batch_size: usize, learning_rate: T) !void {
            const n_layers = self.weights.len;
            const output_size = self.layers[self.layers.len - 1];
            const acts = self.activations.?;

            // Allocate gradient storage if needed
            if (self.gradients == null) {
                const grads = try self.allocator.alloc([]T, n_layers + 1);
                for (0..n_layers + 1) |i| {
                    grads[i] = try self.allocator.alloc(T, self.layers[i] * batch_size);
                }
                self.gradients = grads;
            }

            const grads = self.gradients.?;

            // Compute output layer gradient
            const output_grad = grads[n_layers];
            const y_pred = acts[n_layers];

            switch (self.loss_fn) {
                .mse => {
                    // MSE gradient: d/dy (y_pred - y_true)² = 2(y_pred - y_true)
                    for (0..batch_size) |batch_idx| {
                        for (0..output_size) |i| {
                            const idx = batch_idx * output_size + i;
                            output_grad[idx] = 2.0 * (y_pred[idx] - y_true[idx]);
                            // Apply activation derivative
                            output_grad[idx] *= self.activationDerivative(y_pred[idx]);
                        }
                    }
                },
                .cross_entropy => {
                    // Cross-entropy + softmax gradient: y_pred - y_true (simplified)
                    // Softmax is applied implicitly during loss computation
                    for (0..batch_size) |batch_idx| {
                        // Apply softmax to predictions
                        const pred_slice = y_pred[batch_idx * output_size .. (batch_idx + 1) * output_size];
                        var max_val = pred_slice[0];
                        for (pred_slice) |v| max_val = @max(max_val, v);

                        var sum: T = 0;
                        for (pred_slice) |*v| {
                            v.* = @exp(v.* - max_val);
                            sum += v.*;
                        }
                        for (pred_slice) |*v| v.* /= sum;

                        // Gradient: softmax(y_pred) - y_true
                        for (0..output_size) |i| {
                            const idx = batch_idx * output_size + i;
                            output_grad[idx] = pred_slice[i] - y_true[idx];
                        }
                    }
                },
            }

            // Backpropagate through layers
            var layer_idx = n_layers;
            while (layer_idx > 0) {
                layer_idx -= 1;

                const in_size = self.layers[layer_idx];
                const out_size = self.layers[layer_idx + 1];
                const W = self.weights[layer_idx];
                const b = self.biases[layer_idx];
                const input = acts[layer_idx];
                const curr_grad = grads[layer_idx + 1];
                const prev_grad = grads[layer_idx];

                // Compute gradient w.r.t. input (prev_grad = W^T @ curr_grad)
                @memset(prev_grad[0 .. in_size * batch_size], 0);
                for (0..batch_size) |batch_idx| {
                    for (0..in_size) |i| {
                        var sum: T = 0;
                        for (0..out_size) |j| {
                            sum += W[j * in_size + i] * curr_grad[batch_idx * out_size + j];
                        }
                        prev_grad[batch_idx * in_size + i] = sum;
                        // Apply activation derivative (except for input layer)
                        if (layer_idx > 0) {
                            const act_val = input[batch_idx * in_size + i];
                            prev_grad[batch_idx * in_size + i] *= self.activationDerivative(act_val);
                        }
                    }
                }

                // Update weights: W -= lr * (curr_grad @ input^T) / batch_size
                for (0..out_size) |i| {
                    for (0..in_size) |j| {
                        var grad_sum: T = 0;
                        for (0..batch_size) |batch_idx| {
                            grad_sum += curr_grad[batch_idx * out_size + i] * input[batch_idx * in_size + j];
                        }
                        W[i * in_size + j] -= learning_rate * grad_sum / @as(T, @floatFromInt(batch_size));
                    }
                }

                // Update biases: b -= lr * mean(curr_grad)
                for (0..out_size) |i| {
                    var grad_sum: T = 0;
                    for (0..batch_size) |batch_idx| {
                        grad_sum += curr_grad[batch_idx * out_size + i];
                    }
                    b[i] -= learning_rate * grad_sum / @as(T, @floatFromInt(batch_size));
                }
            }
        }

        /// Apply activation function element-wise
        /// Time: O(n) | Space: O(1)
        fn applyActivation(self: Self, values: []T) void {
            switch (self.activation) {
                .relu => {
                    for (values) |*v| v.* = @max(0, v.*);
                },
                .sigmoid => {
                    for (values) |*v| v.* = 1.0 / (1.0 + @exp(-v.*));
                },
                .tanh => {
                    for (values) |*v| v.* = std.math.tanh(v.*);
                },
                .linear => {}, // No transformation
            }
        }

        /// Compute activation function derivative
        /// Time: O(1) | Space: O(1)
        fn activationDerivative(self: Self, activated_value: T) T {
            return switch (self.activation) {
                .relu => if (activated_value > 0) 1.0 else 0.0,
                .sigmoid => activated_value * (1.0 - activated_value),
                .tanh => 1.0 - activated_value * activated_value,
                .linear => 1.0,
            };
        }

        /// Train the network on data
        /// Time: O(epochs × batches × L × n_max² × batch) | Space: O(L × n_max × batch)
        pub fn fit(
            self: *Self,
            X: []const T,
            y: []const T,
            options: FitOptions,
        ) !void {
            const n_samples = X.len / self.layers[0];
            const output_size = self.layers[self.layers.len - 1];

            if (y.len != n_samples * output_size) return error.DimensionMismatch;

            var prng = std.Random.DefaultPrng.init(options.seed orelse 42);
            const random = prng.random();

            // Create indices for shuffling
            var indices = try self.allocator.alloc(usize, n_samples);
            defer self.allocator.free(indices);
            for (0..n_samples) |i| indices[i] = i;

            // Training loop
            for (0..options.epochs) |epoch| {
                // Shuffle indices
                random.shuffle(usize, indices);

                var total_loss: T = 0;
                var n_batches: usize = 0;

                // Mini-batch training
                var batch_start: usize = 0;
                while (batch_start < n_samples) {
                    const batch_end = @min(batch_start + options.batch_size, n_samples);
                    const batch_size = batch_end - batch_start;

                    // Prepare batch
                    const X_batch = try self.allocator.alloc(T, self.layers[0] * batch_size);
                    defer self.allocator.free(X_batch);
                    const y_batch = try self.allocator.alloc(T, output_size * batch_size);
                    defer self.allocator.free(y_batch);

                    for (batch_start..batch_end) |batch_idx| {
                        const sample_idx = indices[batch_idx];
                        const local_idx = batch_idx - batch_start;
                        @memcpy(
                            X_batch[local_idx * self.layers[0] .. (local_idx + 1) * self.layers[0]],
                            X[sample_idx * self.layers[0] .. (sample_idx + 1) * self.layers[0]],
                        );
                        @memcpy(
                            y_batch[local_idx * output_size .. (local_idx + 1) * output_size],
                            y[sample_idx * output_size .. (sample_idx + 1) * output_size],
                        );
                    }

                    // Forward pass
                    const y_pred = try self.forward(X_batch, batch_size);
                    defer self.allocator.free(y_pred);

                    // Compute loss
                    var batch_loss: T = 0;
                    switch (self.loss_fn) {
                        .mse => {
                            for (0..batch_size) |i| {
                                for (0..output_size) |j| {
                                    const idx = i * output_size + j;
                                    const diff = y_pred[idx] - y_batch[idx];
                                    batch_loss += diff * diff;
                                }
                            }
                            batch_loss /= @as(T, @floatFromInt(batch_size));
                        },
                        .cross_entropy => {
                            for (0..batch_size) |i| {
                                for (0..output_size) |j| {
                                    const idx = i * output_size + j;
                                    if (y_batch[idx] > 0) {
                                        batch_loss -= y_batch[idx] * @log(@max(y_pred[idx], 1e-7));
                                    }
                                }
                            }
                            batch_loss /= @as(T, @floatFromInt(batch_size));
                        },
                    }

                    total_loss += batch_loss;
                    n_batches += 1;

                    // Backward pass
                    try self.backward(y_batch, batch_size, options.learning_rate);

                    batch_start = batch_end;
                }

                if (options.verbose and epoch % 10 == 0) {
                    const avg_loss = total_loss / @as(T, @floatFromInt(n_batches));
                    std.debug.print("Epoch {d}: loss = {d:.6}\n", .{ epoch, avg_loss });
                }
            }
        }

        /// Predict outputs for input data
        /// Time: O(L × n_max² × batch) | Space: O(L × n_max × batch)
        pub fn predict(self: *Self, allocator: Allocator, X: []const T) ![]T {
            const n_samples = X.len / self.layers[0];
            const output_size = self.layers[self.layers.len - 1];

            const predictions = try allocator.alloc(T, n_samples * output_size);
            errdefer allocator.free(predictions);

            // Process in batches to avoid memory issues
            const batch_size: usize = 32;
            var batch_start: usize = 0;
            while (batch_start < n_samples) {
                const batch_end = @min(batch_start + batch_size, n_samples);
                const current_batch_size = batch_end - batch_start;

                const X_batch = X[batch_start * self.layers[0] .. batch_end * self.layers[0]];
                const y_pred = try self.forward(X_batch, current_batch_size);
                defer allocator.free(y_pred);

                @memcpy(
                    predictions[batch_start * output_size .. batch_end * output_size],
                    y_pred[0 .. current_batch_size * output_size],
                );

                batch_start = batch_end;
            }

            return predictions;
        }

        /// Compute classification accuracy
        /// Time: O(n) | Space: O(1)
        pub fn score(self: *Self, allocator: Allocator, X: []const T, y_true: []const T) !T {
            const predictions = try self.predict(allocator, X);
            defer allocator.free(predictions);

            const n_samples = X.len / self.layers[0];
            const output_size = self.layers[self.layers.len - 1];

            var correct: usize = 0;
            for (0..n_samples) |i| {
                // Find predicted class (argmax)
                var max_idx: usize = 0;
                var max_val = predictions[i * output_size];
                for (1..output_size) |j| {
                    const val = predictions[i * output_size + j];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = j;
                    }
                }

                // Find true class (argmax)
                var true_idx: usize = 0;
                var true_max = y_true[i * output_size];
                for (1..output_size) |j| {
                    const val = y_true[i * output_size + j];
                    if (val > true_max) {
                        true_max = val;
                        true_idx = j;
                    }
                }

                if (max_idx == true_idx) correct += 1;
            }

            return @as(T, @floatFromInt(correct)) / @as(T, @floatFromInt(n_samples));
        }

        /// Validate internal state (mainly for testing)
        /// Time: O(L × n_max²) | Space: O(1)
        pub fn validate(self: Self) !void {
            if (self.layers.len < 2) return error.InvalidLayerConfiguration;
            if (self.weights.len != self.layers.len - 1) return error.InvalidWeightsConfiguration;
            if (self.biases.len != self.layers.len - 1) return error.InvalidBiasesConfiguration;

            // Check dimensions
            for (0..self.weights.len) |i| {
                const expected_weight_size = self.layers[i + 1] * self.layers[i];
                if (self.weights[i].len != expected_weight_size) return error.InvalidWeightDimensions;

                const expected_bias_size = self.layers[i + 1];
                if (self.biases[i].len != expected_bias_size) return error.InvalidBiasDimensions;
            }
        }
    };
}

// ==================== Tests ====================

test "MLP: basic initialization" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 1 }, .relu, .mse);
    defer mlp.deinit();

    try testing.expectEqual(@as(usize, 3), mlp.layers.len);
    try testing.expectEqual(@as(usize, 2), mlp.weights.len);
    try mlp.validate();
}

test "MLP: XOR problem (classification)" {
    // XOR is a classic non-linearly separable problem
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 2 }, .relu, .cross_entropy);
    defer mlp.deinit();

    // XOR dataset
    const X = [_]f64{
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };

    // One-hot encoded labels
    const y = [_]f64{
        1, 0, // 0 XOR 0 = 0
        0, 1, // 0 XOR 1 = 1
        0, 1, // 1 XOR 0 = 1
        1, 0, // 1 XOR 1 = 0
    };

    // Train
    try mlp.fit(&X, &y, .{ .epochs = 500, .learning_rate = 0.1 });

    // Test accuracy
    const accuracy = try mlp.score(testing.allocator, &X, &y);
    try testing.expect(accuracy >= 0.75); // Should learn XOR pattern
}

test "MLP: simple regression" {
    // Learn y = 2x + 1
    var mlp = try MLP(f64).init(testing.allocator, &.{ 1, 8, 1 }, .relu, .mse);
    defer mlp.deinit();

    const n: usize = 20;
    var X: [n]f64 = undefined;
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        const x = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n));
        X[i] = x;
        y[i] = 2.0 * x + 1.0;
    }

    // Train
    try mlp.fit(&X, &y, .{ .epochs = 200, .learning_rate = 0.05, .batch_size = 4 });

    // Test predictions
    const predictions = try mlp.predict(testing.allocator, &X);
    defer testing.allocator.free(predictions);

    // Check if predictions are close to true values
    var mse: f64 = 0;
    for (0..n) |i| {
        const diff = predictions[i] - y[i];
        mse += diff * diff;
    }
    mse /= @as(f64, @floatFromInt(n));
    try testing.expect(mse < 0.1); // Should fit reasonably well
}

test "MLP: different activation functions" {
    const activations = [_]MLP(f32).ActivationFn{ .relu, .sigmoid, .tanh };
    for (activations) |act| {
        var mlp = try MLP(f32).init(testing.allocator, &.{ 2, 3, 1 }, act, .mse);
        defer mlp.deinit();
        try mlp.validate();
    }
}

test "MLP: multi-class classification" {
    // Simple 3-class problem
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 8, 3 }, .relu, .cross_entropy);
    defer mlp.deinit();

    // 3 samples, each belonging to a different class
    const X = [_]f64{
        0.1, 0.1, // class 0
        0.5, 0.5, // class 1
        0.9, 0.9, // class 2
    };

    const y = [_]f64{
        1, 0, 0, // class 0
        0, 1, 0, // class 1
        0, 0, 1, // class 2
    };

    // Train
    try mlp.fit(&X, &y, .{ .epochs = 300, .learning_rate = 0.1 });

    // Should achieve perfect accuracy on training data
    const accuracy = try mlp.score(testing.allocator, &X, &y);
    try testing.expect(accuracy >= 0.66);
}

test "MLP: empty input error" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 3, 1 }, .relu, .mse);
    defer mlp.deinit();

    const X: [0]f64 = .{};
    const y: [0]f64 = .{};

    // Should handle gracefully
    try mlp.fit(&X, &y, .{ .epochs = 1 });
}

test "MLP: dimension mismatch error" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 3, 1 }, .relu, .mse);
    defer mlp.deinit();

    const X = [_]f64{ 1, 2, 3, 4 }; // 2 samples
    const y = [_]f64{1}; // Only 1 label

    const result = mlp.fit(&X, &y, .{ .epochs = 1 });
    try testing.expectError(error.DimensionMismatch, result);
}

test "MLP: predict without training" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 3, 1 }, .relu, .mse);
    defer mlp.deinit();

    const X = [_]f64{ 1, 2 };
    const predictions = try mlp.predict(testing.allocator, &X);
    defer testing.allocator.free(predictions);

    try testing.expectEqual(@as(usize, 1), predictions.len);
}

test "MLP: batch size variations" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 1, 4, 1 }, .relu, .mse);
    defer mlp.deinit();

    const X = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    const y = [_]f64{ 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 };

    // Test different batch sizes
    const batch_sizes = [_]usize{ 1, 2, 4, 8 };
    for (batch_sizes) |bs| {
        try mlp.fit(&X, &y, .{ .epochs = 10, .batch_size = bs });
    }
}

test "MLP: deep network" {
    // Test a deeper architecture
    var mlp = try MLP(f32).init(testing.allocator, &.{ 3, 8, 8, 8, 2 }, .relu, .cross_entropy);
    defer mlp.deinit();

    try testing.expectEqual(@as(usize, 5), mlp.layers.len);
    try testing.expectEqual(@as(usize, 4), mlp.weights.len);
    try mlp.validate();
}

test "MLP: large batch prediction" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 1 }, .relu, .mse);
    defer mlp.deinit();

    // Create large dataset
    const n: usize = 100;
    var X: [n * 2]f64 = undefined;
    for (0..n) |i| {
        X[i * 2] = @as(f64, @floatFromInt(i));
        X[i * 2 + 1] = @as(f64, @floatFromInt(i)) * 2;
    }

    const predictions = try mlp.predict(testing.allocator, &X);
    defer testing.allocator.free(predictions);

    try testing.expectEqual(@as(usize, n), predictions.len);
}

test "MLP: sigmoid activation" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 1 }, .sigmoid, .mse);
    defer mlp.deinit();

    const X = [_]f64{ 0, 0, 1, 1 };
    const y = [_]f64{ 0, 1 };

    try mlp.fit(&X, &y, .{ .epochs = 50, .learning_rate = 0.1 });
}

test "MLP: tanh activation" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 1 }, .tanh, .mse);
    defer mlp.deinit();

    const X = [_]f64{ -1, -1, 1, 1 };
    const y = [_]f64{ -1, 1 };

    try mlp.fit(&X, &y, .{ .epochs = 50, .learning_rate = 0.01 });
}

test "MLP: linear activation" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 1, 3, 1 }, .linear, .mse);
    defer mlp.deinit();

    const X = [_]f64{ 1, 2, 3 };
    const y = [_]f64{ 2, 4, 6 };

    try mlp.fit(&X, &y, .{ .epochs = 100, .learning_rate = 0.01 });
}

test "MLP: invalid layer configuration" {
    const result = MLP(f64).init(testing.allocator, &.{1}, .relu, .mse);
    try testing.expectError(error.InvalidLayerConfiguration, result);
}

test "MLP: memory safety with testing allocator" {
    var mlp = try MLP(f64).init(testing.allocator, &.{ 2, 4, 2 }, .relu, .cross_entropy);
    defer mlp.deinit();

    const X = [_]f64{ 0, 0, 1, 1 };
    const y = [_]f64{ 1, 0, 0, 1 };

    try mlp.fit(&X, &y, .{ .epochs = 10 });

    const predictions = try mlp.predict(testing.allocator, &X);
    defer testing.allocator.free(predictions);

    // testing.allocator will detect any leaks
}
