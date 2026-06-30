const std = @import("std");
const math = std.math;

pub fn main() void {
    const c: f64 = 0.5;
    const sigma: f64 = 1.0;
    const k = @sqrt(c);
    const sigma_sq = sigma * sigma;
    const lambda_minus = sigma_sq * (1.0 - k) * (1.0 - k);
    const lambda_plus = sigma_sq * (1.0 + k) * (1.0 + k);
    
    std.debug.print("lambda_minus = {d}, lambda_plus = {d}\n", .{lambda_minus, lambda_plus});
    
    var max_pdf: f64 = 0.0;
    var max_x: f64 = 0.0;
    
    var x: f64 = lambda_minus + 0.001;
    while (x < lambda_plus) {
        const pdf_num = @sqrt((lambda_plus - x) * (x - lambda_minus));
        const pdf_denom = 2.0 * math.pi * c * sigma_sq * x;
        const pdf_val = pdf_num / pdf_denom;
        if (pdf_val > max_pdf) {
            max_pdf = pdf_val;
            max_x = x;
        }
        x += 0.001;
    }
    
    std.debug.print("Maximum PDF at x = {d}, pdf = {d}\n", .{max_x, max_pdf});
    
    const mid = (lambda_minus + lambda_plus) / 2.0;
    const pdf_mid_num = @sqrt((lambda_plus - mid) * (mid - lambda_minus));
    const pdf_mid_denom = 2.0 * math.pi * c * sigma_sq * mid;
    const pdf_mid = pdf_mid_num / pdf_mid_denom;
    
    std.debug.print("PDF at midpoint = {d}\n", .{pdf_mid});
}
