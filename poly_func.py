import numpy as np
from scipy.optimize import leastsq

# Reference points (pixel coordinates)
pixel_coords = np.array([
    [141, 110],
    [1308, 37],
    [136, 1388],
    [1438, 1385]
])

# Corresponding GPS coordinates
gps_coords = np.array([
    [30.283116, -97.753163],
    [30.283238, -97.750912],
    [30.280984, -97.753174],
    [30.280983, -97.750658]
])

# Affine function to fit
def affine_func(p, xy):
    x, y = xy
    lat = p[0] + p[1]*x + p[2]*y
    lon = p[3] + p[4]*x + p[5]*y
    return np.vstack((lat, lon))

# Residuals function for least squares optimization
def residuals(p, xy, gps):
    return (affine_func(p, xy) - gps).ravel()

# Initial guess for the parameters
p0 = np.zeros(6)

# Solve for the affine coefficients
xy = pixel_coords.T
gps = gps_coords.T
p_opt, _ = leastsq(residuals, p0, args=(xy, gps))

# Transformation function
def pixel_to_gps_affine(x, y, p):
    return affine_func(p, (x, y))

# Haversine formula to calculate distance between two GPS coordinates in meters
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# Example usage and error calculation
checkpoints = [
    {"pixel": (1160, 1266), "gps": (30.281193, -97.751193)},
    {"pixel": (1206, 1394), "gps": (30.280973, -97.751108)},
    {"pixel": (844, 494), "gps": (30.282474, -97.751810)}
    # Add more checkpoints as needed
]

for checkpoint in checkpoints:
    x, y = checkpoint["pixel"]
    true_lat, true_lon = checkpoint["gps"]
    est_lat, est_lon = pixel_to_gps_affine(x, y, p_opt)
    lat_error = est_lat - true_lat
    lon_error = est_lon - true_lon
    distance_error = haversine(true_lat, true_lon, est_lat[0], est_lon[0])
    
    print(f"Pixel coordinates: ({x}, {y})")
    print(f"True GPS coordinates: Latitude = {true_lat}, Longitude = {true_lon}")
    print(f"Estimated GPS coordinates: Latitude = {est_lat[0]:.6f}, Longitude = {est_lon[0]:.6f}")
    print(f"Latitude error: {lat_error[0]} degrees")
    print(f"Longitude error: {lon_error[0]} degrees")
    print(f"Distance error: {distance_error} meters")
    print()