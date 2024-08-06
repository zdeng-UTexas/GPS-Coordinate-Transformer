import numpy as np

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

# Construct the matrices for affine transformation
A = np.vstack([pixel_coords.T, np.ones(4)]).T
B = gps_coords

# Solve for transformation coefficients
coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

# Transformation function
def pixel_to_gps(x, y, coefficients):
    gps = np.dot(coefficients.T, np.array([x, y, 1]))
    return gps

# Example usage
x, y = 1160, 1266  # Pixel coordinates
lat, lon = pixel_to_gps(x, y, coefficients)
print(f"GPS coordinates: Latitude = {lat}, Longitude = {lon}")
