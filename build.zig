const std = @import("std");

// Build configuration for zuda v1.19.0 — Matrix Decompositions (LU, QR, Cholesky, SVD, Eigendecomposition)
// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("zuda", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "zuda",
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "zuda" is the name you will use in your source code to
                // import this module (e.g. `@import("zuda")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Scientific workflow example demonstrating cross-module integration
    const scientific_example = b.addExecutable(.{
        .name = "scientific_workflow",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/scientific_workflow.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_scientific_example = b.addRunArtifact(scientific_example);
    const example_step = b.step("example", "Run the scientific workflow example");
    example_step.dependOn(&run_scientific_example.step);
    run_scientific_example.step.dependOn(b.getInstallStep());

    // Machine learning pipeline example
    const ml_example = b.addExecutable(.{
        .name = "ml_pipeline",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ml_pipeline.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_ml_example = b.addRunArtifact(ml_example);
    const ml_example_step = b.step("example-ml", "Run the machine learning pipeline example");
    ml_example_step.dependOn(&run_ml_example.step);
    run_ml_example.step.dependOn(b.getInstallStep());

    // Time series analysis example
    const timeseries_example = b.addExecutable(.{
        .name = "timeseries_analysis",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/timeseries_analysis.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_timeseries_example = b.addRunArtifact(timeseries_example);
    const timeseries_example_step = b.step("example-timeseries", "Run the time series analysis example");
    timeseries_example_step.dependOn(&run_timeseries_example.step);
    run_timeseries_example.step.dependOn(b.getInstallStep());

    // Physics simulation example
    const physics_example = b.addExecutable(.{
        .name = "physics_simulation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/physics_simulation.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_physics_example = b.addRunArtifact(physics_example);
    const physics_example_step = b.step("example-physics", "Run the physics simulation example");
    physics_example_step.dependOn(&run_physics_example.step);
    run_physics_example.step.dependOn(b.getInstallStep());

    // Optimization showcase example
    const optimization_example = b.addExecutable(.{
        .name = "optimization_showcase",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/optimization_showcase.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_optimization_example = b.addRunArtifact(optimization_example);
    const optimization_example_step = b.step("example-optimization", "Run the optimization showcase example");
    optimization_example_step.dependOn(&run_optimization_example.step);
    run_optimization_example.step.dependOn(b.getInstallStep());

    // Neural network example
    const nn_example = b.addExecutable(.{
        .name = "neural_network",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/neural_network.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_nn_example = b.addRunArtifact(nn_example);
    const nn_example_step = b.step("example-nn", "Run the neural network training example");
    nn_example_step.dependOn(&run_nn_example.step);
    run_nn_example.step.dependOn(b.getInstallStep());

    // Image processing example
    const image_example = b.addExecutable(.{
        .name = "image_processing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/image_processing.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_image_example = b.addRunArtifact(image_example);
    const image_example_step = b.step("example-image", "Run the image processing example");
    image_example_step.dependOn(&run_image_example.step);
    run_image_example.step.dependOn(b.getInstallStep());

    // Monte Carlo simulation example
    const monte_carlo_example = b.addExecutable(.{
        .name = "monte_carlo_simulation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/monte_carlo_simulation.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_monte_carlo_example = b.addRunArtifact(monte_carlo_example);
    const monte_carlo_example_step = b.step("example-montecarlo", "Run the Monte Carlo simulation example");
    monte_carlo_example_step.dependOn(&run_monte_carlo_example.step);
    run_monte_carlo_example.step.dependOn(b.getInstallStep());

    // PDE solver example
    const pde_example = b.addExecutable(.{
        .name = "pde_solver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/pde_solver.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_pde_example = b.addRunArtifact(pde_example);
    const pde_example_step = b.step("example-pde", "Run the PDE solver example");
    pde_example_step.dependOn(&run_pde_example.step);
    run_pde_example.step.dependOn(b.getInstallStep());

    // Computational geometry example
    const geometry_example = b.addExecutable(.{
        .name = "computational_geometry",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/computational_geometry.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_geometry_example = b.addRunArtifact(geometry_example);
    const geometry_example_step = b.step("example-geometry", "Run the computational geometry example");
    geometry_example_step.dependOn(&run_geometry_example.step);
    run_geometry_example.step.dependOn(b.getInstallStep());

    // K-Means clustering example
    const clustering_example = b.addExecutable(.{
        .name = "clustering",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/clustering.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_clustering_example = b.addRunArtifact(clustering_example);
    const clustering_example_step = b.step("example-clustering", "Run the K-Means clustering example");
    clustering_example_step.dependOn(&run_clustering_example.step);
    run_clustering_example.step.dependOn(b.getInstallStep());

    // Kalman filter example
    const kalman_example = b.addExecutable(.{
        .name = "kalman_filter",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/kalman_filter.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(kalman_example);
    const run_kalman_example = b.addRunArtifact(kalman_example);
    const kalman_example_step = b.step("example-kalman", "Run the Kalman filter example");
    kalman_example_step.dependOn(&run_kalman_example.step);
    run_kalman_example.step.dependOn(b.getInstallStep());

    // Anomaly detection example
    const anomaly_example = b.addExecutable(.{
        .name = "anomaly_detection",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/anomaly_detection.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(anomaly_example);
    const run_anomaly_example = b.addRunArtifact(anomaly_example);
    const anomaly_example_step = b.step("example-anomaly", "Run the anomaly detection example");
    anomaly_example_step.dependOn(&run_anomaly_example.step);
    run_anomaly_example.step.dependOn(b.getInstallStep());

    // Signal processing example
    const signal_example = b.addExecutable(.{
        .name = "signal_processing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/signal_processing.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(signal_example);
    const run_signal_example = b.addRunArtifact(signal_example);
    const signal_example_step = b.step("example-signal", "Run the signal processing example");
    signal_example_step.dependOn(&run_signal_example.step);
    run_signal_example.step.dependOn(b.getInstallStep());

    // Financial Modeling Example
    const financial_example = b.addExecutable(.{
        .name = "financial_modeling",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/financial_modeling.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(financial_example);
    const run_financial_example = b.addRunArtifact(financial_example);
    const financial_example_step = b.step("example-financial", "Run the financial modeling example");
    financial_example_step.dependOn(&run_financial_example.step);
    run_financial_example.step.dependOn(b.getInstallStep());

    // Bayesian Inference Example
    const bayesian_example = b.addExecutable(.{
        .name = "bayesian_inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bayesian_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(bayesian_example);
    const run_bayesian_example = b.addRunArtifact(bayesian_example);
    const bayesian_example_step = b.step("example-bayesian", "Run the Bayesian inference example");
    bayesian_example_step.dependOn(&run_bayesian_example.step);
    run_bayesian_example.step.dependOn(b.getInstallStep());

    // Control systems example
    const control_example = b.addExecutable(.{
        .name = "control_systems",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/control_systems.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(control_example);
    const run_control_example = b.addRunArtifact(control_example);
    const control_example_step = b.step("example-control", "Run the control systems example");
    control_example_step.dependOn(&run_control_example.step);
    run_control_example.step.dependOn(b.getInstallStep());

    // Stochastic processes example
    const stochastic_example = b.addExecutable(.{
        .name = "stochastic_processes",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/stochastic_processes.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    b.installArtifact(stochastic_example);
    const run_stochastic_example = b.addRunArtifact(stochastic_example);
    const stochastic_example_step = b.step("example-stochastic", "Run the stochastic processes example");
    stochastic_example_step.dependOn(&run_stochastic_example.step);
    run_stochastic_example.step.dependOn(b.getInstallStep());

    // Shared library with C API for FFI
    const shared = b.option(bool, "shared", "Build shared library with C API") orelse false;
    if (shared) {
        const lib = b.addLibrary(.{
            .name = "zuda",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/ffi/c_api.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        lib.linkLibC();
        b.installArtifact(lib);

        // Install C header
        const install_header = b.addInstallFileWithDir(
            b.path("include/zuda.h"),
            .header,
            "zuda.h",
        );
        b.getInstallStep().dependOn(&install_header.step);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // Memory safety audit tests
    const memory_safety_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/memory_safety_audit.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_memory_safety_tests = b.addRunArtifact(memory_safety_tests);

    // Cross-module integration tests (Phase 12)
    const cross_module_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/cross_module_integration.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });
    const run_cross_module_tests = b.addRunArtifact(cross_module_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_memory_safety_tests.step);
    test_step.dependOn(&run_cross_module_tests.step);

    // Separate step for memory safety audit
    const memory_safety_step = b.step("test-memory", "Run memory safety audit tests");
    memory_safety_step.dependOn(&run_memory_safety_tests.step);

    // Separate step for cross-module integration tests
    const cross_module_step = b.step("test-integration", "Run cross-module integration tests");
    cross_module_step.dependOn(&run_cross_module_tests.step);

    // Benchmark executables
    const bench_trees = b.addExecutable(.{
        .name = "bench_trees",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/trees.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_heaps = b.addExecutable(.{
        .name = "bench_heaps",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/heaps.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_btrees = b.addExecutable(.{
        .name = "bench_btrees",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/btrees.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_probabilistic = b.addExecutable(.{
        .name = "bench_probabilistic",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/probabilistic.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_graphs = b.addExecutable(.{
        .name = "bench_graphs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/graphs.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_sorting = b.addExecutable(.{
        .name = "bench_sorting",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/sorting.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_strings = b.addExecutable(.{
        .name = "bench_strings",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/strings.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_memory = b.addExecutable(.{
        .name = "bench_memory",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/memory_profile.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_memory_strings = b.addExecutable(.{
        .name = "bench_memory_strings",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/memory_strings.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_rbtree_micro = b.addExecutable(.{
        .name = "bench_rbtree_micro",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/rbtree_micro.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_lists = b.addExecutable(.{
        .name = "bench_lists",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/lists.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_queues = b.addExecutable(.{
        .name = "bench_queues",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/queues.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_hashing = b.addExecutable(.{
        .name = "bench_hashing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/hashing.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_cache = b.addExecutable(.{
        .name = "bench_cache",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/cache.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_cache_profile = b.addExecutable(.{
        .name = "bench_cache_profile",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/cache_profile_strings.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    const bench_scientific = b.addExecutable(.{
        .name = "bench_scientific",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/scientific_computing.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zuda", .module = mod },
            },
        }),
    });

    // Install benchmark executables
    b.installArtifact(bench_trees);
    b.installArtifact(bench_heaps);
    b.installArtifact(bench_btrees);
    b.installArtifact(bench_probabilistic);
    b.installArtifact(bench_graphs);
    b.installArtifact(bench_sorting);
    b.installArtifact(bench_strings);
    b.installArtifact(bench_memory);
    b.installArtifact(bench_memory_strings);
    b.installArtifact(bench_rbtree_micro);
    b.installArtifact(bench_lists);
    b.installArtifact(bench_queues);
    b.installArtifact(bench_hashing);
    b.installArtifact(bench_cache);
    b.installArtifact(bench_cache_profile);
    b.installArtifact(bench_scientific);

    const bench_step = b.step("bench", "Run benchmarks");
    const run_bench_trees = b.addRunArtifact(bench_trees);
    const run_bench_heaps = b.addRunArtifact(bench_heaps);
    const run_bench_btrees = b.addRunArtifact(bench_btrees);
    const run_bench_probabilistic = b.addRunArtifact(bench_probabilistic);
    const run_bench_graphs = b.addRunArtifact(bench_graphs);
    const run_bench_sorting = b.addRunArtifact(bench_sorting);
    const run_bench_strings = b.addRunArtifact(bench_strings);
    const run_bench_lists = b.addRunArtifact(bench_lists);
    const run_bench_queues = b.addRunArtifact(bench_queues);
    const run_bench_hashing = b.addRunArtifact(bench_hashing);
    const run_bench_cache = b.addRunArtifact(bench_cache);

    bench_step.dependOn(&run_bench_trees.step);
    bench_step.dependOn(&run_bench_heaps.step);
    bench_step.dependOn(&run_bench_btrees.step);
    bench_step.dependOn(&run_bench_probabilistic.step);
    bench_step.dependOn(&run_bench_graphs.step);
    bench_step.dependOn(&run_bench_sorting.step);
    bench_step.dependOn(&run_bench_strings.step);
    bench_step.dependOn(&run_bench_lists.step);
    bench_step.dependOn(&run_bench_queues.step);
    bench_step.dependOn(&run_bench_hashing.step);
    bench_step.dependOn(&run_bench_cache.step);

    run_bench_trees.step.dependOn(b.getInstallStep());
    run_bench_heaps.step.dependOn(b.getInstallStep());
    run_bench_btrees.step.dependOn(b.getInstallStep());
    run_bench_probabilistic.step.dependOn(b.getInstallStep());
    run_bench_graphs.step.dependOn(b.getInstallStep());
    run_bench_sorting.step.dependOn(b.getInstallStep());
    run_bench_strings.step.dependOn(b.getInstallStep());
    run_bench_lists.step.dependOn(b.getInstallStep());
    run_bench_queues.step.dependOn(b.getInstallStep());
    run_bench_hashing.step.dependOn(b.getInstallStep());
    run_bench_cache.step.dependOn(b.getInstallStep());

    // Memory profiling benchmark (separate step)
    const bench_memory_step = b.step("bench-memory", "Run memory profiling benchmarks");
    const run_bench_memory = b.addRunArtifact(bench_memory);
    bench_memory_step.dependOn(&run_bench_memory.step);
    run_bench_memory.step.dependOn(b.getInstallStep());

    // Memory profiling for string structures (v1.8.0 validation)
    const bench_memory_strings_step = b.step("bench-memory-strings", "Run memory profiling for string structures");
    const run_bench_memory_strings = b.addRunArtifact(bench_memory_strings);
    bench_memory_strings_step.dependOn(&run_bench_memory_strings.step);
    run_bench_memory_strings.step.dependOn(b.getInstallStep());

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
