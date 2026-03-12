//! Haversine formula for calculating great-circle distances between two points on a sphere.
//!
//! The haversine formula determines the shortest distance over the earth's surface,
//! giving an "as-the-crow-flies" distance between two points (ignoring elevation,
//! terrain, roads, etc.).
//!
//! Time complexity: O(1)
//! Space complexity: O(1)

const std = @import("std");
const math = std.math;

/// Default Earth radius in kilometers
pub const EARTH_RADIUS_KM: f64 = 6371.0;

/// Default Earth radius in miles
pub const EARTH_RADIUS_MI: f64 = 3959.0;

/// Default Earth radius in meters
pub const EARTH_RADIUS_M: f64 = 6371000.0;

/// Coordinate in degrees (latitude, longitude)
pub const Coord = struct {
    lat: f64,
    lon: f64,

    /// Create a coordinate from degrees
    /// Time: O(1) | Space: O(1)
    pub fn init(lat: f64, lon: f64) Coord {
        return .{ .lat = lat, .lon = lon };
    }

    /// Validate that coordinates are within valid ranges
    /// Latitude: [-90, 90], Longitude: [-180, 180]
    /// Time: O(1) | Space: O(1)
    pub fn isValid(self: Coord) bool {
        return self.lat >= -90.0 and self.lat <= 90.0 and
            self.lon >= -180.0 and self.lon <= 180.0;
    }

    /// Convert degrees to radians
    /// Time: O(1) | Space: O(1)
    pub fn toRadians(degrees: f64) f64 {
        return degrees * math.pi / 180.0;
    }
};

/// Calculate the haversine distance between two coordinates using the specified radius.
/// Time: O(1) | Space: O(1)
///
/// The haversine formula:
/// a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
/// c = 2 · atan2(√a, √(1−a))
/// d = R · c
///
/// Arguments:
/// - from: starting coordinate (lat, lon in degrees)
/// - to: ending coordinate (lat, lon in degrees)
/// - radius: sphere radius in desired units (km, mi, m, etc.)
///
/// Returns: distance in the same units as radius
pub fn distance(from: Coord, to: Coord, radius: f64) f64 {
    const lat1 = Coord.toRadians(from.lat);
    const lon1 = Coord.toRadians(from.lon);
    const lat2 = Coord.toRadians(to.lat);
    const lon2 = Coord.toRadians(to.lon);

    const dlat = lat2 - lat1;
    const dlon = lon2 - lon1;

    const a = math.pow(f64, math.sin(dlat / 2.0), 2.0) +
        math.cos(lat1) * math.cos(lat2) *
        math.pow(f64, math.sin(dlon / 2.0), 2.0);

    const c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a));

    return radius * c;
}

/// Calculate the haversine distance in kilometers.
/// Time: O(1) | Space: O(1)
pub fn distanceKm(from: Coord, to: Coord) f64 {
    return distance(from, to, EARTH_RADIUS_KM);
}

/// Calculate the haversine distance in miles.
/// Time: O(1) | Space: O(1)
pub fn distanceMi(from: Coord, to: Coord) f64 {
    return distance(from, to, EARTH_RADIUS_MI);
}

/// Calculate the haversine distance in meters.
/// Time: O(1) | Space: O(1)
pub fn distanceM(from: Coord, to: Coord) f64 {
    return distance(from, to, EARTH_RADIUS_M);
}

/// Calculate the initial bearing (forward azimuth) from one point to another.
/// The bearing is the angle measured clockwise from true north.
/// Time: O(1) | Space: O(1)
///
/// Returns: bearing in degrees [0, 360)
pub fn initialBearing(from: Coord, to: Coord) f64 {
    const lat1 = Coord.toRadians(from.lat);
    const lon1 = Coord.toRadians(from.lon);
    const lat2 = Coord.toRadians(to.lat);
    const lon2 = Coord.toRadians(to.lon);

    const dlon = lon2 - lon1;

    const y = math.sin(dlon) * math.cos(lat2);
    const x = math.cos(lat1) * math.sin(lat2) -
        math.sin(lat1) * math.cos(lat2) * math.cos(dlon);

    const bearing_rad = math.atan2(y, x);
    const bearing_deg = bearing_rad * 180.0 / math.pi;

    // Normalize to [0, 360)
    return @mod(bearing_deg + 360.0, 360.0);
}

/// Calculate the destination point given a start point, bearing, and distance.
/// Time: O(1) | Space: O(1)
///
/// Arguments:
/// - from: starting coordinate
/// - bearing: initial bearing in degrees
/// - distance_val: distance to travel
/// - radius: sphere radius in same units as distance_val
///
/// Returns: destination coordinate
pub fn destination(from: Coord, bearing: f64, distance_val: f64, radius: f64) Coord {
    const lat1 = Coord.toRadians(from.lat);
    const lon1 = Coord.toRadians(from.lon);
    const brng = Coord.toRadians(bearing);
    const d_r = distance_val / radius; // angular distance

    const lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r) +
            math.cos(lat1) * math.sin(d_r) * math.cos(brng),
    );

    const lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    );

    return Coord{
        .lat = lat2 * 180.0 / math.pi,
        .lon = lon2 * 180.0 / math.pi,
    };
}

/// Calculate the midpoint between two coordinates.
/// Time: O(1) | Space: O(1)
pub fn midpoint(from: Coord, to: Coord) Coord {
    const lat1 = Coord.toRadians(from.lat);
    const lon1 = Coord.toRadians(from.lon);
    const lat2 = Coord.toRadians(to.lat);
    const lon2 = Coord.toRadians(to.lon);

    const dlon = lon2 - lon1;

    const bx = math.cos(lat2) * math.cos(dlon);
    const by = math.cos(lat2) * math.sin(dlon);

    const lat3 = math.atan2(
        math.sin(lat1) + math.sin(lat2),
        math.sqrt((math.cos(lat1) + bx) * (math.cos(lat1) + bx) + by * by),
    );
    const lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx);

    return Coord{
        .lat = lat3 * 180.0 / math.pi,
        .lon = lon3 * 180.0 / math.pi,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "haversine: basic distance calculation" {
    const testing = std.testing;

    // New York to London (approximate)
    const ny = Coord.init(40.7128, -74.0060);
    const london = Coord.init(51.5074, -0.1278);

    const dist_km = distanceKm(ny, london);

    // Expected distance is approximately 5570 km
    try testing.expect(dist_km > 5500.0 and dist_km < 5600.0);
}

test "haversine: same point distance is zero" {
    const testing = std.testing;

    const coord = Coord.init(37.7749, -122.4194); // San Francisco
    const dist = distanceKm(coord, coord);

    try testing.expectApproxEqAbs(0.0, dist, 0.001);
}

test "haversine: equator points" {
    const testing = std.testing;

    const p1 = Coord.init(0.0, 0.0);
    const p2 = Coord.init(0.0, 1.0);

    const dist_km = distanceKm(p1, p2);

    // 1 degree at equator ≈ 111.32 km
    try testing.expectApproxEqAbs(111.32, dist_km, 1.0);
}

test "haversine: different radius units" {
    const testing = std.testing;

    const p1 = Coord.init(37.7749, -122.4194); // San Francisco
    const p2 = Coord.init(34.0522, -118.2437); // Los Angeles

    const dist_km = distanceKm(p1, p2);
    const dist_mi = distanceMi(p1, p2);
    const dist_m = distanceM(p1, p2);

    // Verify conversions
    try testing.expectApproxEqAbs(dist_km, dist_m / 1000.0, 1.0);
    try testing.expectApproxEqAbs(dist_mi, dist_km * 0.621371, 1.0);

    // SF to LA is approximately 560 km / 347 mi
    try testing.expect(dist_km > 550.0 and dist_km < 570.0);
    try testing.expect(dist_mi > 340.0 and dist_mi < 360.0);
}

test "haversine: coordinate validation" {
    const testing = std.testing;

    const valid = Coord.init(45.0, -90.0);
    try testing.expect(valid.isValid());

    const invalid_lat = Coord.init(91.0, 0.0);
    try testing.expect(!invalid_lat.isValid());

    const invalid_lon = Coord.init(0.0, 181.0);
    try testing.expect(!invalid_lon.isValid());

    const edge_valid = Coord.init(-90.0, -180.0);
    try testing.expect(edge_valid.isValid());
}

test "haversine: initial bearing calculation" {
    const testing = std.testing;

    // San Francisco to New York (approximately east)
    const sf = Coord.init(37.7749, -122.4194);
    const ny = Coord.init(40.7128, -74.0060);

    const bearing = initialBearing(sf, ny);

    // Bearing should be roughly east-northeast (60-90 degrees)
    try testing.expect(bearing > 60.0 and bearing < 90.0);
}

test "haversine: bearing range normalization" {
    const testing = std.testing;

    const p1 = Coord.init(0.0, 0.0);
    const p2 = Coord.init(0.0, -1.0);

    const bearing = initialBearing(p1, p2);

    // Bearing west should be 270 degrees
    try testing.expectApproxEqAbs(270.0, bearing, 1.0);
}

test "haversine: destination calculation" {
    const testing = std.testing;

    const start = Coord.init(37.7749, -122.4194); // San Francisco
    const bearing_val = 90.0; // Due east
    const distance_val = 100.0; // 100 km

    const dest = destination(start, bearing_val, distance_val, EARTH_RADIUS_KM);

    // Verify destination is valid
    try testing.expect(dest.isValid());

    // Verify we moved approximately 100 km
    const actual_dist = distanceKm(start, dest);
    try testing.expectApproxEqAbs(distance_val, actual_dist, 1.0);

    // Verify we moved east (longitude increased)
    try testing.expect(dest.lon > start.lon);

    // Latitude should be similar (moving along a latitude line)
    try testing.expectApproxEqAbs(start.lat, dest.lat, 1.0);
}

test "haversine: round trip destination" {
    const testing = std.testing;

    const start = Coord.init(51.5074, -0.1278); // London
    const bearing_val = 45.0; // Northeast
    const distance_val = 100.0; // 100 km (shorter distance for better accuracy)

    // Go forward
    const dest = destination(start, bearing_val, distance_val, EARTH_RADIUS_KM);

    // Calculate the actual bearing from dest back to start
    const actual_back_bearing = initialBearing(dest, start);

    // Go back using the actual bearing
    const back = destination(dest, actual_back_bearing, distance_val, EARTH_RADIUS_KM);

    // Should be close to start (within 5 km due to spherical geometry)
    const error_dist = distanceKm(start, back);
    try testing.expect(error_dist < 5.0);
}

test "haversine: midpoint calculation" {
    const testing = std.testing;

    const p1 = Coord.init(0.0, 0.0);
    const p2 = Coord.init(0.0, 10.0);

    const mid = midpoint(p1, p2);

    // Midpoint along equator
    try testing.expectApproxEqAbs(0.0, mid.lat, 0.001);
    try testing.expectApproxEqAbs(5.0, mid.lon, 0.001);
}

test "haversine: midpoint symmetry" {
    const testing = std.testing;

    const sf = Coord.init(37.7749, -122.4194);
    const ny = Coord.init(40.7128, -74.0060);

    const mid1 = midpoint(sf, ny);
    const mid2 = midpoint(ny, sf);

    // Midpoint should be the same regardless of order
    try testing.expectApproxEqAbs(mid1.lat, mid2.lat, 0.001);
    try testing.expectApproxEqAbs(mid1.lon, mid2.lon, 0.001);

    // Distance from each endpoint to midpoint should be equal
    const dist1 = distanceKm(sf, mid1);
    const dist2 = distanceKm(ny, mid1);
    try testing.expectApproxEqAbs(dist1, dist2, 1.0);
}

test "haversine: antipodal points" {
    const testing = std.testing;

    // Points on opposite sides of Earth
    const p1 = Coord.init(0.0, 0.0);
    const p2 = Coord.init(0.0, 180.0);

    const dist = distanceKm(p1, p2);

    // Distance should be approximately half Earth's circumference
    // Circumference = 2πr ≈ 40,030 km, half ≈ 20,015 km
    const half_circumference = math.pi * EARTH_RADIUS_KM;
    try testing.expectApproxEqAbs(half_circumference, dist, 10.0);
}

test "haversine: north and south pole" {
    const testing = std.testing;

    const north = Coord.init(90.0, 0.0);
    const south = Coord.init(-90.0, 0.0);

    const dist = distanceKm(north, south);

    // Distance should be approximately half Earth's circumference
    const half_circumference = math.pi * EARTH_RADIUS_KM;
    try testing.expectApproxEqAbs(half_circumference, dist, 10.0);
}

test "haversine: stress test with random coordinates" {
    const testing = std.testing;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const lat1 = random.float(f64) * 180.0 - 90.0;
        const lon1 = random.float(f64) * 360.0 - 180.0;
        const lat2 = random.float(f64) * 180.0 - 90.0;
        const lon2 = random.float(f64) * 360.0 - 180.0;

        const p1 = Coord.init(lat1, lon1);
        const p2 = Coord.init(lat2, lon2);

        try testing.expect(p1.isValid());
        try testing.expect(p2.isValid());

        const dist = distanceKm(p1, p2);

        // Distance should be non-negative and less than half Earth's circumference
        try testing.expect(dist >= 0.0);
        try testing.expect(dist <= math.pi * EARTH_RADIUS_KM + 1.0);

        // Distance should be symmetric
        const dist_reverse = distanceKm(p2, p1);
        try testing.expectApproxEqAbs(dist, dist_reverse, 0.001);
    }
}
