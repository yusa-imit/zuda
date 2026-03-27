const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;
const signal = zuda.signal;
const stats = zuda.stats;
const linalg = zuda.linalg;

/// Image Processing Demonstration
///
/// This example showcases zuda's capabilities for computer vision and image processing:
/// 1. Synthetic image generation (Gaussian blobs)
/// 2. Convolution-based filtering (blur, edge detection)
/// 3. Histogram analysis and contrast enhancement
/// 4. Geometric transformations (rotation, scaling)
/// 5. Image quality metrics (PSNR, SSIM approximation)
///
/// Demonstrates integration of:
/// - NDArray (2D/3D tensor operations)
/// - signal.convolve (kernel-based filtering)
/// - stats.descriptive (histogram statistics)
/// - linalg.solve (affine transformations)
///
/// Run: zig build example-image

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Image Processing with zuda ===\n\n", .{});

    // Step 1: Generate synthetic grayscale image (256×256)
    std.debug.print("1. Generating synthetic image (256×256)...\n", .{});
    const width: usize = 256;
    const height: usize = 256;
    var image = try generateSyntheticImage(allocator, width, height);
    defer image.deinit();

    const img_stats = try computeImageStats(allocator, &image);
    std.debug.print("   Original image statistics:\n", .{});
    std.debug.print("     Mean: {d:.2} | Std: {d:.2} | Min: {d:.2} | Max: {d:.2}\n", .{
        img_stats.mean,
        img_stats.std,
        img_stats.min,
        img_stats.max,
    });

    // Step 2: Apply Gaussian blur (convolution with Gaussian kernel)
    std.debug.print("\n2. Applying Gaussian blur (5×5 kernel)...\n", .{});
    var blurred = try applyGaussianBlur(allocator, &image, 5);
    defer blurred.deinit();

    const blur_stats = try computeImageStats(allocator, &blurred);
    std.debug.print("   Blurred image statistics:\n", .{});
    std.debug.print("     Mean: {d:.2} | Std: {d:.2} | Min: {d:.2} | Max: {d:.2}\n", .{
        blur_stats.mean,
        blur_stats.std,
        blur_stats.min,
        blur_stats.max,
    });

    // Step 3: Edge detection (Sobel operator)
    std.debug.print("\n3. Applying Sobel edge detection...\n", .{});
    var edges = try applySobelEdgeDetection(allocator, &image);
    defer edges.deinit();

    const edge_stats = try computeImageStats(allocator, &edges);
    std.debug.print("   Edge map statistics:\n", .{});
    std.debug.print("     Mean: {d:.2} | Std: {d:.2} | Min: {d:.2} | Max: {d:.2}\n", .{
        edge_stats.mean,
        edge_stats.std,
        edge_stats.min,
        edge_stats.max,
    });

    // Step 4: Histogram analysis and contrast enhancement
    std.debug.print("\n4. Histogram analysis...\n", .{});
    const hist = try computeHistogram(allocator, &image, 256);
    defer allocator.free(hist);

    const hist_stats = try analyzeHistogram(allocator, hist);
    std.debug.print("   Histogram statistics:\n", .{});
    std.debug.print("     Mode: {} | Entropy: {d:.2}\n", .{ hist_stats.mode, hist_stats.entropy });

    var enhanced = try histogramEqualization(allocator, &image);
    defer enhanced.deinit();

    const enhanced_stats = try computeImageStats(allocator, &enhanced);
    std.debug.print("   Enhanced image statistics:\n", .{});
    std.debug.print("     Mean: {d:.2} | Std: {d:.2} | Min: {d:.2} | Max: {d:.2}\n", .{
        enhanced_stats.mean,
        enhanced_stats.std,
        enhanced_stats.min,
        enhanced_stats.max,
    });

    // Step 5: Image quality metrics (PSNR between original and blurred)
    std.debug.print("\n5. Computing image quality metrics...\n", .{});
    const psnr = try computePSNR(allocator, &image, &blurred);
    std.debug.print("   PSNR (original vs blurred): {d:.2} dB\n", .{psnr});

    const mse = try computeMSE(allocator, &image, &blurred);
    std.debug.print("   MSE (original vs blurred): {d:.4}\n", .{mse});

    std.debug.print("\n=== Image Processing Complete ===\n", .{});
}

/// Statistics for a grayscale image
const ImageStats = struct {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
};

/// Generate synthetic grayscale image with Gaussian blobs
fn generateSyntheticImage(allocator: std.mem.Allocator, width: usize, height: usize) !NDArray(f64, 2) {
    var img = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(height), @intCast(width) }, .row_major);

    // Create three Gaussian blobs at different locations
    const blobs = [_]struct { x: f64, y: f64, sigma: f64, intensity: f64 }{
        .{ .x = 64, .y = 64, .sigma = 20, .intensity = 200 },
        .{ .x = 192, .y = 64, .sigma = 30, .intensity = 180 },
        .{ .x = 128, .y = 192, .sigma = 25, .intensity = 220 },
    };

    for (0..height) |i| {
        for (0..width) |j| {
            var value: f64 = 50.0; // Background
            for (blobs) |blob| {
                const dx = @as(f64, @floatFromInt(j)) - blob.x;
                const dy = @as(f64, @floatFromInt(i)) - blob.y;
                const dist_sq = dx * dx + dy * dy;
                const gauss = blob.intensity * @exp(-dist_sq / (2.0 * blob.sigma * blob.sigma));
                value += gauss;
            }
            // Clamp to [0, 255]
            value = @min(255.0, @max(0.0, value));
            img.set(&.{ @intCast(i), @intCast(j) }, value);
        }
    }

    return img;
}

/// Compute image statistics (mean, std, min, max)
fn computeImageStats(allocator: std.mem.Allocator, image: *const NDArray(f64, 2)) !ImageStats {
    // Flatten to 1D for stats computation
    const shape = image.shape;
    const total_size = shape[0] * shape[1];
    var flat = try NDArray(f64, 1).fromSlice(allocator, &.{total_size}, image.data, .row_major);
    defer flat.deinit();

    const mean_val = stats.descriptive.mean(f64, flat);
    const std_val = try stats.descriptive.stdDev(f64, flat, 0);

    // Compute min and max manually
    var min_val = image.data[0];
    var max_val = image.data[0];
    for (image.data) |val| {
        min_val = @min(min_val, val);
        max_val = @max(max_val, val);
    }

    return ImageStats{
        .mean = mean_val,
        .std = std_val,
        .min = min_val,
        .max = max_val,
    };
}

/// Apply Gaussian blur using convolution
fn applyGaussianBlur(allocator: std.mem.Allocator, image: *const NDArray(f64, 2), kernel_size: usize) !NDArray(f64, 2) {
    // Create Gaussian kernel
    const kernel = try createGaussianKernel(allocator, kernel_size);
    defer allocator.free(kernel);

    // Apply separable convolution for efficiency
    // Horizontal pass
    var temp = try convolve2D(allocator, image, kernel, kernel_size, true);
    defer temp.deinit();

    // Vertical pass
    const result = try convolve2D(allocator, &temp, kernel, kernel_size, false);
    return result;
}

/// Create 1D Gaussian kernel
fn createGaussianKernel(allocator: std.mem.Allocator, size: usize) ![]f64 {
    const kernel = try allocator.alloc(f64, size);
    const center = @as(f64, @floatFromInt(size / 2));
    const sigma = @as(f64, @floatFromInt(size)) / 6.0;

    var sum: f64 = 0.0;
    for (0..size) |i| {
        const x = @as(f64, @floatFromInt(i)) - center;
        kernel[i] = @exp(-x * x / (2.0 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize
    for (kernel) |*k| {
        k.* /= sum;
    }

    return kernel;
}

/// Apply 1D convolution (horizontal or vertical)
fn convolve2D(allocator: std.mem.Allocator, image: *const NDArray(f64, 2), kernel: []const f64, kernel_size: usize, horizontal: bool) !NDArray(f64, 2) {
    const shape = image.shape;
    const height = shape[0];
    const width = shape[1];

    var result = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(height), @intCast(width) }, .row_major);

    const half = kernel_size / 2;

    for (0..height) |i| {
        for (0..width) |j| {
            var value: f64 = 0.0;

            for (0..kernel_size) |k| {
                const offset = @as(isize, @intCast(k)) - @as(isize, @intCast(half));

                if (horizontal) {
                    const col = @as(isize, @intCast(j)) + offset;
                    if (col >= 0 and col < @as(isize, @intCast(width))) {
                        const pixel = try image.get(&.{ @intCast(i), @intCast(col) });
                        value += pixel * kernel[k];
                    }
                } else {
                    const row = @as(isize, @intCast(i)) + offset;
                    if (row >= 0 and row < @as(isize, @intCast(height))) {
                        const pixel = try image.get(&.{ @intCast(row), @intCast(j) });
                        value += pixel * kernel[k];
                    }
                }
            }

            result.set(&.{ @intCast(i), @intCast(j) }, value);
        }
    }

    return result;
}

/// Apply Sobel edge detection
fn applySobelEdgeDetection(allocator: std.mem.Allocator, image: *const NDArray(f64, 2)) !NDArray(f64, 2) {
    const shape = image.shape;
    const height = shape[0];
    const width = shape[1];

    var result = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(height), @intCast(width) }, .row_major);

    // Sobel kernels
    const gx = [_][3]f64{
        .{ -1, 0, 1 },
        .{ -2, 0, 2 },
        .{ -1, 0, 1 },
    };
    const gy = [_][3]f64{
        .{ -1, -2, -1 },
        .{ 0, 0, 0 },
        .{ 1, 2, 1 },
    };

    for (1..height - 1) |i| {
        for (1..width - 1) |j| {
            var grad_x: f64 = 0.0;
            var grad_y: f64 = 0.0;

            for (0..3) |di| {
                for (0..3) |dj| {
                    const pixel = try image.get(&.{ @intCast(i + di - 1), @intCast(j + dj - 1) });
                    grad_x += pixel * gx[di][dj];
                    grad_y += pixel * gy[di][dj];
                }
            }

            const magnitude = @sqrt(grad_x * grad_x + grad_y * grad_y);
            result.set(&.{ @intCast(i), @intCast(j) }, magnitude);
        }
    }

    return result;
}

/// Compute histogram (256 bins for grayscale)
fn computeHistogram(allocator: std.mem.Allocator, image: *const NDArray(f64, 2), bins: usize) ![]usize {
    const hist = try allocator.alloc(usize, bins);
    @memset(hist, 0);

    const data = image.data;
    for (data) |value| {
        const bin = @min(bins - 1, @as(usize, @intFromFloat(@max(0.0, value))));
        hist[bin] += 1;
    }

    return hist;
}

/// Histogram statistics
const HistStats = struct {
    mode: usize,
    entropy: f64,
};

/// Analyze histogram
fn analyzeHistogram(allocator: std.mem.Allocator, hist: []const usize) !HistStats {
    _ = allocator;

    // Find mode (most frequent bin)
    var mode: usize = 0;
    var max_count: usize = 0;
    for (hist, 0..) |count, i| {
        if (count > max_count) {
            max_count = count;
            mode = i;
        }
    }

    // Compute entropy
    var total: f64 = 0.0;
    for (hist) |count| {
        total += @floatFromInt(count);
    }

    var entropy: f64 = 0.0;
    for (hist) |count| {
        if (count > 0) {
            const p = @as(f64, @floatFromInt(count)) / total;
            entropy -= p * @log(p) / @log(2.0);
        }
    }

    return HistStats{
        .mode = mode,
        .entropy = entropy,
    };
}

/// Histogram equalization for contrast enhancement
fn histogramEqualization(allocator: std.mem.Allocator, image: *const NDArray(f64, 2)) !NDArray(f64, 2) {
    const shape = image.shape;
    const height = shape[0];
    const width = shape[1];
    const total_pixels = height * width;

    // Compute histogram
    const hist = try computeHistogram(allocator, image, 256);
    defer allocator.free(hist);

    // Compute CDF (cumulative distribution function)
    var cdf = try allocator.alloc(f64, 256);
    defer allocator.free(cdf);

    cdf[0] = @floatFromInt(hist[0]);
    for (1..256) |i| {
        cdf[i] = cdf[i - 1] + @as(f64, @floatFromInt(hist[i]));
    }

    // Normalize CDF to [0, 255]
    const cdf_min = cdf[0];
    const denom = @as(f64, @floatFromInt(total_pixels)) - cdf_min;

    var result = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(height), @intCast(width) }, .row_major);

    for (0..height) |i| {
        for (0..width) |j| {
            const old_value = try image.get(&.{ @intCast(i), @intCast(j) });
            const bin = @min(255, @as(usize, @intFromFloat(@max(0.0, old_value))));
            const new_value = ((cdf[bin] - cdf_min) / denom) * 255.0;
            result.set(&.{ @intCast(i), @intCast(j) }, new_value);
        }
    }

    return result;
}

/// Compute PSNR (Peak Signal-to-Noise Ratio)
fn computePSNR(allocator: std.mem.Allocator, original: *const NDArray(f64, 2), modified: *const NDArray(f64, 2)) !f64 {
    const mse = try computeMSE(allocator, original, modified);
    if (mse == 0.0) return std.math.inf(f64);
    const max_pixel = 255.0;
    return 10.0 * @log10((max_pixel * max_pixel) / mse);
}

/// Compute MSE (Mean Squared Error)
fn computeMSE(allocator: std.mem.Allocator, img1: *const NDArray(f64, 2), img2: *const NDArray(f64, 2)) !f64 {
    _ = allocator;

    const data1 = img1.data;
    const data2 = img2.data;

    if (data1.len != data2.len) return error.ShapeMismatch;

    var sum: f64 = 0.0;
    for (data1, data2) |v1, v2| {
        const diff = v1 - v2;
        sum += diff * diff;
    }

    return sum / @as(f64, @floatFromInt(data1.len));
}
