import numpy as np

def read_labeled_points(file_path):
    points = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            px, py, lat, lon = map(float, line.strip().split())
            points.append((px, py, lat, lon))
    return points

def compute_affine_transformation(points):
    A = []
    B_lat = []
    B_lon = []

    for (px, py, lat, lon) in points:
        A.append([px, py, 1, 0, 0, 0])
        A.append([0, 0, 0, px, py, 1])
        B_lat.append(lat)
        B_lat.append(0)
        B_lon.append(0)
        B_lon.append(lon)

    A = np.array(A)
    B_lat = np.array(B_lat)
    B_lon = np.array(B_lon)

    coefficients_lat = np.linalg.lstsq(A, B_lat, rcond=None)[0]
    coefficients_lon = np.linalg.lstsq(A, B_lon, rcond=None)[0]

    return coefficients_lat, coefficients_lon

def estimate_gps(px, py, coefficients_lat, coefficients_lon):
    a, b, c, _, _, _ = coefficients_lat
    _, _, _, d, e, f = coefficients_lon
    lat = a * px + b * py + c
    lon = d * px + e * py + f
    return lat, lon

def main(image_size, input_file_path, output_file_path):
    width, height = map(int, image_size.split('x'))
    points = read_labeled_points(input_file_path)
    
    if len(points) < 3:
        print("Need at least 3 points to estimate the transformation.")
        return

    coefficients_lat, coefficients_lon = compute_affine_transformation(points)

    with open(output_file_path, 'w') as output_file:
        for px in range(width):
            for py in range(height):
                lat, lon = estimate_gps(px, py, coefficients_lat, coefficients_lon)
                output_file.write(f"{px} {py} {lat} {lon}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Estimate GPS coordinates for each pixel in an image.')
    parser.add_argument('image_size', type=str, help='The pixel size of the image in the format widthxheight (e.g., 1497x1497)')
    parser.add_argument('input_file_path', type=str, help='Path to the text file containing labeled points (format: px py lat lon)')
    parser.add_argument('output_file_path', type=str, help='Path to the output text file to save the estimated GPS coordinates')

    args = parser.parse_args()
    main(args.image_size, args.input_file_path, args.output_file_path)
