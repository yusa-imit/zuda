const std = @import("std");
const zuda = @import("zuda");

// Import geometry functions
const geometry = zuda.algorithms.geometry;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Computational Geometry Demonstration ===\n\n", .{});

    // Part 1: Geohash Encoding & Spatial Indexing
    std.debug.print("Part 1: Geohash Spatial Indexing\n", .{});
    std.debug.print("--------------------------------\n", .{});
    try demonstrateGeohash();

    // Part 2: Haversine Distance & Proximity Search
    std.debug.print("\nPart 2: Haversine Distance Calculation\n", .{});
    std.debug.print("---------------------------------------\n", .{});
    try demonstrateHaversine();

    // Part 3: Convex Hull for Location Clustering
    std.debug.print("\nPart 3: Convex Hull Computation\n", .{});
    std.debug.print("--------------------------------\n", .{});
    try demonstrateConvexHull(allocator);

    // Part 4: Integrated Geospatial Application
    std.debug.print("\nPart 4: Integrated Geospatial Analysis\n", .{});
    std.debug.print("---------------------------------------\n", .{});
    try demonstrateIntegratedAnalysis();

    std.debug.print("\n=== Demonstration Complete ===\n", .{});
}

/// Demonstrate geohash encoding and decoding for spatial indexing
fn demonstrateGeohash() !void {
    // Example locations: Major cities
    const locations = [_]struct { name: []const u8, lat: f64, lon: f64 }{
        .{ .name = "San Francisco", .lat = 37.7749, .lon = -122.4194 },
        .{ .name = "New York", .lat = 40.7128, .lon = -74.0060 },
        .{ .name = "London", .lat = 51.5074, .lon = -0.1278 },
        .{ .name = "Tokyo", .lat = 35.6762, .lon = 139.6503 },
        .{ .name = "Sydney", .lat = -33.8688, .lon = 151.2093 },
    };

    std.debug.print("Encoding city locations to geohashes:\n", .{});
    for (locations) |loc| {
        // Encode with precision 8 (±19m accuracy)
        var buffer: [12]u8 = undefined;
        const hash = try geometry.geohashEncode(loc.lat, loc.lon, 8, &buffer);

        // Decode back to verify
        const bbox = try geometry.geohashDecode(hash);
        const center = bbox.center();

        std.debug.print("  {s:15} ({d:8.4}, {d:9.4}) → geohash: {s} → ({d:8.4}, {d:9.4})\n", .{
            loc.name,
            loc.lat,
            loc.lon,
            hash,
            center.lat,
            center.lon,
        });
    }

    // Demonstrate prefix-based proximity
    std.debug.print("\nGeohash prefix similarity (longer prefix = closer locations):\n", .{});
    var sf_buffer: [12]u8 = undefined;
    var oakland_buffer: [12]u8 = undefined;
    var la_buffer: [12]u8 = undefined;

    const sf_hash = try geometry.geohashEncode(37.7749, -122.4194, 8, &sf_buffer); // San Francisco
    const oakland_hash = try geometry.geohashEncode(37.8044, -122.2712, 8, &oakland_buffer); // Oakland (nearby)
    const la_hash = try geometry.geohashEncode(34.0522, -118.2437, 8, &la_buffer); // Los Angeles (far)

    std.debug.print("  San Francisco: {s}\n", .{sf_hash});
    std.debug.print("  Oakland:       {s} (shares {d} prefix chars)\n", .{ oakland_hash, commonPrefixLength(sf_hash, oakland_hash) });
    std.debug.print("  Los Angeles:   {s} (shares {d} prefix chars)\n", .{ la_hash, commonPrefixLength(sf_hash, la_hash) });
}

/// Demonstrate haversine distance calculation
fn demonstrateHaversine() !void {
    const Coord = geometry.Coord;
    const EARTH_RADIUS_KM = 6371.0;

    // Calculate distances between major cities
    const sf = Coord.init(37.7749, -122.4194); // San Francisco
    const ny = Coord.init(40.7128, -74.0060); // New York
    const london = Coord.init(51.5074, -0.1278); // London
    const tokyo = Coord.init(35.6762, 139.6503); // Tokyo

    std.debug.print("Great-circle distances between cities:\n", .{});

    const sf_to_ny = geometry.haversineDistance(sf, ny, EARTH_RADIUS_KM);
    std.debug.print("  San Francisco → New York:   {d:8.1} km\n", .{sf_to_ny});

    const sf_to_london = geometry.haversineDistance(sf, london, EARTH_RADIUS_KM);
    std.debug.print("  San Francisco → London:     {d:8.1} km\n", .{sf_to_london});

    const sf_to_tokyo = geometry.haversineDistance(sf, tokyo, EARTH_RADIUS_KM);
    std.debug.print("  San Francisco → Tokyo:      {d:8.1} km\n", .{sf_to_tokyo});

    const ny_to_london = geometry.haversineDistance(ny, london, EARTH_RADIUS_KM);
    std.debug.print("  New York → London:          {d:8.1} km\n", .{ny_to_london});

    // Proximity search: find cities within 10000km of San Francisco
    const radius_km = 10000.0;
    std.debug.print("\nCities within {d:.0} km of San Francisco:\n", .{radius_km});

    const cities = [_]struct { name: []const u8, coord: Coord }{
        .{ .name = "New York", .coord = ny },
        .{ .name = "London", .coord = london },
        .{ .name = "Tokyo", .coord = tokyo },
    };

    for (cities) |city| {
        const dist = geometry.haversineDistance(sf, city.coord, EARTH_RADIUS_KM);
        if (dist <= radius_km) {
            std.debug.print("  ✓ {s:12} ({d:8.1} km)\n", .{ city.name, dist });
        } else {
            std.debug.print("  ✗ {s:12} ({d:8.1} km) - too far\n", .{ city.name, dist });
        }
    }
}

/// Demonstrate convex hull computation for location clustering
fn demonstrateConvexHull(allocator: std.mem.Allocator) !void {
    const Point = geometry.Point;

    // Simulate warehouse locations in a distribution center
    const warehouses = [_]Point(f64){
        Point(f64).init(2.0, 3.0),
        Point(f64).init(4.0, 8.0),
        Point(f64).init(7.0, 5.0),
        Point(f64).init(3.0, 1.0),
        Point(f64).init(8.0, 2.0),
        Point(f64).init(5.0, 6.0),
        Point(f64).init(6.0, 4.0),
        Point(f64).init(1.0, 5.0),
    };

    std.debug.print("Computing convex hull of {d} warehouse locations:\n", .{warehouses.len});
    std.debug.print("Input points: ", .{});
    for (warehouses) |p| {
        std.debug.print("({d:.1}, {d:.1}) ", .{ p.x, p.y });
    }
    std.debug.print("\n", .{});

    // Compute convex hull using Graham scan
    const hull = try geometry.grahamScan(f64, allocator, &warehouses);
    defer allocator.free(hull);

    std.debug.print("Convex hull ({d} vertices):\n", .{hull.len});
    for (hull, 0..) |p, i| {
        std.debug.print("  [{d}] ({d:.1}, {d:.1})\n", .{ i, p.x, p.y });
    }

    // Calculate perimeter of the hull (boundary fence length)
    var perimeter: f64 = 0.0;
    for (0..hull.len) |i| {
        const p1 = hull[i];
        const p2 = hull[(i + 1) % hull.len];
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        perimeter += @sqrt(dx * dx + dy * dy);
    }
    std.debug.print("Perimeter (fence length needed): {d:.2} units\n", .{perimeter});
}

/// Integrated geospatial analysis combining multiple algorithms
fn demonstrateIntegratedAnalysis() !void {
    const Coord = geometry.Coord;
    const EARTH_RADIUS_KM = 6371.0;

    std.debug.print("Scenario: Food delivery service in San Francisco\n", .{});

    // Define restaurant and customer locations (lat, lon)
    const Restaurant = struct {
        name: []const u8,
        coord: Coord,
    };

    const restaurants = [_]Restaurant{
        .{ .name = "Pizza Palace", .coord = Coord.init(37.7749, -122.4194) },
        .{ .name = "Sushi Station", .coord = Coord.init(37.7849, -122.4094) },
        .{ .name = "Burger Barn", .coord = Coord.init(37.7649, -122.4294) },
    };

    const customer = Coord.init(37.7799, -122.4144); // Customer location

    std.debug.print("\nStep 1: Find nearest restaurant using haversine distance\n", .{});
    var min_distance: f64 = std.math.inf(f64);
    var nearest_restaurant: ?Restaurant = null;

    for (restaurants) |restaurant| {
        const dist = geometry.haversineDistance(customer, restaurant.coord, EARTH_RADIUS_KM);
        std.debug.print("  {s:15} distance: {d:6.2} km\n", .{ restaurant.name, dist });

        if (dist < min_distance) {
            min_distance = dist;
            nearest_restaurant = restaurant;
        }
    }

    if (nearest_restaurant) |r| {
        std.debug.print("  → Nearest: {s} ({d:.2} km)\n", .{ r.name, min_distance });
    }

    std.debug.print("\nStep 2: Encode locations to geohashes for spatial indexing\n", .{});
    var customer_buffer: [12]u8 = undefined;
    const customer_hash = try geometry.geohashEncode(customer.lat, customer.lon, 7, &customer_buffer);
    std.debug.print("  Customer geohash: {s}\n", .{customer_hash});

    for (restaurants) |restaurant| {
        var buffer: [12]u8 = undefined;
        const hash = try geometry.geohashEncode(restaurant.coord.lat, restaurant.coord.lon, 7, &buffer);
        const prefix_len = commonPrefixLength(customer_hash, hash);
        std.debug.print("  {s:15} geohash: {s} (prefix match: {d}/7)\n", .{
            restaurant.name,
            hash,
            prefix_len,
        });
    }

    std.debug.print("\n✓ Analysis complete: Route optimized using geometry algorithms!\n", .{});
}

/// Helper function to find common prefix length between two strings
fn commonPrefixLength(a: []const u8, b: []const u8) usize {
    const min_len = @min(a.len, b.len);
    var i: usize = 0;
    while (i < min_len) : (i += 1) {
        if (a[i] != b[i]) break;
    }
    return i;
}
