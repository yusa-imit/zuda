//! Partial Differential Equation (PDE) Solver Example
//!
//! Demonstrates solving the 2D heat equation using finite difference methods:
//! ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
//!
//! This example showcases:
//! - NDArray for 2D spatial grids
//! - Finite difference discretization
//! - Time-stepping with explicit Euler method
//! - Boundary condition handling
//! - Visualization via ASCII heat map
//! - Numerical stability analysis (CFL condition)
//!
//! Physical interpretation: Heat diffusion in a 2D plate with fixed boundary temperatures

const std = @import("std");
const zuda = @import("zuda");

const NDArray = zuda.ndarray.NDArray;

/// Heat equation solver using finite difference method
const HeatEquation2D = struct {
    nx: usize, // Grid points in x
    ny: usize, // Grid points in y
    dx: f64, // Spatial step x
    dy: f64, // Spatial step y
    dt: f64, // Time step
    alpha: f64, // Thermal diffusivity
    u: NDArray(f64, 2), // Current temperature field
    u_next: NDArray(f64, 2), // Next temperature field
    allocator: std.mem.Allocator,

    /// Initialize heat equation solver
    /// Time: O(nx × ny), Space: O(nx × ny)
    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, lx: f64, ly: f64, alpha: f64, dt: f64) !HeatEquation2D {
        const dx = lx / @as(f64, @floatFromInt(nx - 1));
        const dy = ly / @as(f64, @floatFromInt(ny - 1));

        // Check CFL stability condition: dt ≤ (dx²dy²) / (2α(dx² + dy²))
        const cfl_max = (dx * dx * dy * dy) / (2.0 * alpha * (dx * dx + dy * dy));
        if (dt > cfl_max) {
            std.debug.print("Warning: CFL condition violated. dt={d} > max={d}. Solution may be unstable.\n", .{ dt, cfl_max });
        }

        var u = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(nx), @intCast(ny) }, .row_major);
        errdefer u.deinit();

        const u_next = try NDArray(f64, 2).zeros(allocator, &.{ @intCast(nx), @intCast(ny) }, .row_major);

        return HeatEquation2D{
            .nx = nx,
            .ny = ny,
            .dx = dx,
            .dy = dy,
            .dt = dt,
            .alpha = alpha,
            .u = u,
            .u_next = u_next,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HeatEquation2D) void {
        self.u.deinit();
        self.u_next.deinit();
    }

    /// Set initial condition
    /// Time: O(1), Space: O(1)
    pub fn setInitial(self: *HeatEquation2D, i: usize, j: usize, value: f64) void {
        self.u.set(&.{ @intCast(i), @intCast(j) }, value);
    }

    /// Set boundary condition (Dirichlet)
    /// Time: O(nx + ny), Space: O(1)
    pub fn setBoundary(self: *HeatEquation2D, value: f64) void {
        // Top and bottom
        for (0..self.nx) |i| {
            self.u.set(&.{ @intCast(i), 0 }, value);
            self.u.set(&.{ @intCast(i), @intCast(self.ny - 1) }, value);
        }
        // Left and right
        for (0..self.ny) |j| {
            self.u.set(&.{ 0, @intCast(j) }, value);
            self.u.set(&.{ @intCast(self.nx - 1), @intCast(j) }, value);
        }
    }

    /// Perform one time step using explicit Euler method
    /// Time: O(nx × ny), Space: O(1)
    pub fn step(self: *HeatEquation2D) void {
        const rx = self.alpha * self.dt / (self.dx * self.dx);
        const ry = self.alpha * self.dt / (self.dy * self.dy);

        // Interior points: five-point stencil
        for (1..self.nx - 1) |i| {
            for (1..self.ny - 1) |j| {
                const u_ij = self.u.get(&.{ @intCast(i), @intCast(j) }) catch 0.0;
                const u_ip1 = self.u.get(&.{ @intCast(i + 1), @intCast(j) }) catch 0.0;
                const u_im1 = self.u.get(&.{ @intCast(i - 1), @intCast(j) }) catch 0.0;
                const u_jp1 = self.u.get(&.{ @intCast(i), @intCast(j + 1) }) catch 0.0;
                const u_jm1 = self.u.get(&.{ @intCast(i), @intCast(j - 1) }) catch 0.0;

                // Finite difference: u_next = u + α×dt×(∂²u/∂x² + ∂²u/∂y²)
                const laplacian = rx * (u_ip1 - 2.0 * u_ij + u_im1) +
                    ry * (u_jp1 - 2.0 * u_ij + u_jm1);
                const u_new = u_ij + laplacian;

                self.u_next.set(&.{ @intCast(i), @intCast(j) }, u_new);
            }
        }

        // Copy boundary conditions
        for (0..self.nx) |i| {
            self.u_next.set(&.{ @intCast(i), 0 }, self.u.get(&.{ @intCast(i), 0 }) catch 0.0);
            self.u_next.set(&.{ @intCast(i), @intCast(self.ny - 1) }, self.u.get(&.{ @intCast(i), @intCast(self.ny - 1) }) catch 0.0);
        }
        for (0..self.ny) |j| {
            self.u_next.set(&.{ 0, @intCast(j) }, self.u.get(&.{ 0, @intCast(j) }) catch 0.0);
            self.u_next.set(&.{ @intCast(self.nx - 1), @intCast(j) }, self.u.get(&.{ @intCast(self.nx - 1), @intCast(j) }) catch 0.0);
        }

        // Swap buffers
        const tmp = self.u;
        self.u = self.u_next;
        self.u_next = tmp;
    }

    /// Compute total energy (integral of u²)
    /// Time: O(nx × ny), Space: O(1)
    pub fn energy(self: *const HeatEquation2D) f64 {
        var total: f64 = 0.0;
        for (0..self.nx) |i| {
            for (0..self.ny) |j| {
                const val = self.u.get(&.{ @intCast(i), @intCast(j) }) catch 0.0;
                total += val * val;
            }
        }
        return total * self.dx * self.dy;
    }

    /// Print ASCII heat map
    /// Time: O(nx × ny), Space: O(1)
    pub fn printHeatMap(self: *const HeatEquation2D) void {
        const chars = " .:-=+*#%@";
        var u_min: f64 = std.math.floatMax(f64);
        var u_max: f64 = -std.math.floatMax(f64);

        // Find min/max
        for (0..self.nx) |i| {
            for (0..self.ny) |j| {
                const val = self.u.get(&.{ @intCast(i), @intCast(j) }) catch 0.0;
                u_min = @min(u_min, val);
                u_max = @max(u_max, val);
            }
        }

        const range = u_max - u_min;
        if (range < 1e-10) {
            std.debug.print("Uniform field: u = {d}\n", .{u_min});
            return;
        }

        std.debug.print("Heat Map (temperature field):\n", .{});
        for (0..self.ny) |j| {
            const jj = self.ny - 1 - j; // Flip y-axis for visualization
            for (0..self.nx) |i| {
                const val = self.u.get(&.{ @intCast(i), @intCast(jj) }) catch 0.0;
                const normalized = (val - u_min) / range;
                const idx: usize = @intFromFloat(@min(normalized * @as(f64, @floatFromInt(chars.len - 1)), @as(f64, @floatFromInt(chars.len - 1))));
                std.debug.print("{c}", .{chars[idx]});
            }
            std.debug.print("\n", .{});
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== 2D Heat Equation Solver ===\n\n", .{});

    // Problem parameters
    const nx: usize = 40; // Grid points x
    const ny: usize = 30; // Grid points y
    const lx: f64 = 1.0; // Domain size x
    const ly: f64 = 1.0; // Domain size y
    const alpha: f64 = 0.01; // Thermal diffusivity
    const dt: f64 = 0.0001; // Time step
    const total_time: f64 = 0.5; // Total simulation time
    const n_steps: usize = @intFromFloat(total_time / dt);
    const print_every: usize = n_steps / 5; // Print 5 snapshots

    std.debug.print("Grid: {d}×{d}, Domain: {d}×{d}\n", .{ nx, ny, lx, ly });
    std.debug.print("Thermal diffusivity α = {d}\n", .{alpha});
    std.debug.print("Time step dt = {d}, Total time = {d}\n", .{ dt, total_time });
    std.debug.print("Steps: {d}, Print every {d} steps\n\n", .{ n_steps, print_every });

    // Initialize solver
    var solver = try HeatEquation2D.init(allocator, nx, ny, lx, ly, alpha, dt);
    defer solver.deinit();

    // Initial condition: hot spot in center
    const center_x = nx / 2;
    const center_y = ny / 2;
    const radius: usize = 5;
    for (0..nx) |i| {
        for (0..ny) |j| {
            const di = @as(isize, @intCast(i)) - @as(isize, @intCast(center_x));
            const dj = @as(isize, @intCast(j)) - @as(isize, @intCast(center_y));
            const dist_sq = di * di + dj * dj;
            if (dist_sq <= @as(isize, @intCast(radius * radius))) {
                solver.setInitial(i, j, 100.0); // Hot spot at 100°
            }
        }
    }

    // Boundary condition: cold boundaries at 0°
    solver.setBoundary(0.0);

    std.debug.print("Initial condition: Hot spot (100°) in center, cold boundaries (0°)\n\n", .{});
    solver.printHeatMap();
    std.debug.print("\nInitial energy: {d}\n\n", .{solver.energy()});

    // Time evolution
    for (0..n_steps) |step| {
        solver.step();

        if ((step + 1) % print_every == 0 or step == n_steps - 1) {
            const time = @as(f64, @floatFromInt(step + 1)) * dt;
            std.debug.print("--- Time t = {d} (step {d}/{d}) ---\n", .{ time, step + 1, n_steps });
            solver.printHeatMap();
            std.debug.print("Energy: {d}\n\n", .{solver.energy()});
        }
    }

    std.debug.print("=== Simulation Complete ===\n", .{});
    std.debug.print("Physical interpretation:\n", .{});
    std.debug.print("- Heat diffuses from hot center to cold boundaries\n", .{});
    std.debug.print("- Energy decreases monotonically (2nd law of thermodynamics)\n", .{});
    std.debug.print("- Final state approaches equilibrium (uniform 0°)\n", .{});
}
