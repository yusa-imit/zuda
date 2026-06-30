const std = @import("std");
const math = std.math;

const DistributionError = error{InvalidParameter, InvalidProbability};

pub fn MarchenkoPastur(comptime T: type) type {
    return struct {
        c: T,
        sigma: T,
        lambda_minus: T,
        lambda_plus: T,

        const Self = @This();

        pub fn init(c: T, sigma: T) DistributionError!Self {
            if (!(c > 0.0 and c <= 1.0) or !math.isFinite(c)) return error.InvalidParameter;
            if (!(sigma > 0.0) or !math.isFinite(sigma)) return error.InvalidParameter;
            const k = @sqrt(c);
            const sigma_sq = sigma * sigma;
            return Self{
                .c = c,
                .sigma = sigma,
                .lambda_minus = sigma_sq * (1.0 - k) * (1.0 - k),
                .lambda_plus = sigma_sq * (1.0 + k) * (1.0 + k),
            };
        }

        pub fn pdf(self: Self, x: T) T {
            if (x <= self.lambda_minus or x >= self.lambda_plus) return 0.0;
            const num = @sqrt((self.lambda_plus - x) * (x - self.lambda_minus));
            const denom = 2.0 * math.pi * self.c * self.sigma * self.sigma * x;
            return num / denom;
        }

        pub fn mean(self: Self) T {
            return self.sigma * self.sigma;
        }

        pub fn variance(self: Self) T {
            return self.c * self.sigma * self.sigma * self.sigma * self.sigma;
        }
    };
}

pub fn main() void {
    const dist = MarchenkoPastur(f64).init(0.5, 1.0) catch unreachable;
    
    std.debug.print("c={d}, sigma={d}\n", .{dist.c, dist.sigma});
    std.debug.print("lambda_minus={d}, lambda_plus={d}\n", .{dist.lambda_minus, dist.lambda_plus});
    std.debug.print("mean={d}, variance={d}\n", .{dist.mean(), dist.variance()});
    std.debug.print("Expected mean={d}, Expected var={d}\n", .{dist.sigma*dist.sigma, dist.c*dist.sigma*dist.sigma*dist.sigma*dist.sigma});
    
    // Test PDF at various points
    const mid = (dist.lambda_minus + dist.lambda_plus) / 2.0;
    const p1 = dist.pdf(dist.lambda_minus + 0.1);
    const pmid = dist.pdf(mid);
    const p2 = dist.pdf(dist.lambda_plus - 0.1);
    
    std.debug.print("\nPDF values:\n", .{});
    std.debug.print("pdf(lambda_minus + 0.1) = {d}\n", .{p1});
    std.debug.print("pdf(midpoint) = {d}\n", .{pmid});
    std.debug.print("pdf(lambda_plus - 0.1) = {d}\n", .{p2});
    std.debug.print("\nTest expectation: pmid >= p1? {}\n", .{pmid >= p1});
}
