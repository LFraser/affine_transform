import numpy as np
import json
import os

def compute_transformation_matrix(source_points, target_points):
    """
    Computes the rigid transformation matrix that aligns source_points to target_points.

    Parameters:
        source_points (numpy.ndarray): Nx3 array of source points.
        target_points (numpy.ndarray): Nx3 array of target points.

    Returns:
        numpy.ndarray: 4x4 transformation matrix.
    """
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have the same shape.")

    # Compute centroids of the point sets
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Handle special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = centroid_target - np.dot(R, centroid_source)

    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    # Compute residuals and RMS
    transformed_points = np.dot(source_points, R.T) + t
    residuals = np.linalg.norm(transformed_points - target_points, axis=1)
    rms = np.sqrt(np.mean(residuals**2))

    print("Residuals:", residuals)
    print("RMS Error:", rms)

    return transformation_matrix

def write_pdal_pipeline(transformation_matrix, input_file, output_file, pipeline_file):
    """
    Writes the transformation matrix to a PDAL JSON pipeline file.

    Parameters:
        transformation_matrix (numpy.ndarray): 4x4 transformation matrix.
        input_file (str): Path to the input LAS file.
        output_file (str): Path to the output LAS file.
        pipeline_file (str): Path to save the PDAL JSON pipeline file.
    """
    matrix_str = " ".join(map(str, transformation_matrix.flatten()))
    pipeline = [
        input_file,
        {
            "type": "filters.transformation",
            "matrix": matrix_str
        },
        {
            "type": "writers.las",
            "filename": output_file
        }
    ]

    with open(pipeline_file, 'w') as f:
        json.dump(pipeline, f, indent=4)

def process_las_files(folder_path, source_points, target_points):
    """
    Processes all .las files in the specified folder, creating a PDAL pipeline for each file.

    Parameters:
        folder_path (str): Path to the folder containing .las files.
        source_points (numpy.ndarray): Nx3 array of source points.
        target_points (numpy.ndarray): Nx3 array of target points.
    """
    # Compute the transformation matrix once
    tform_matrix = compute_transformation_matrix(source_points, target_points)

    # List all .las files in the folder
    las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]

    for las_file in las_files:
        input_file = os.path.join(folder_path, las_file)
        output_file = os.path.join(folder_path, f"{os.path.splitext(las_file)[0]}_transformed.las")
        pipeline_file = os.path.join(folder_path, f"{os.path.splitext(las_file)[0]}_pipeline.json")

        # Write the PDAL pipeline for the current file
        write_pdal_pipeline(tform_matrix, input_file, output_file, pipeline_file)
        print(f"PDAL pipeline written for {las_file} to {pipeline_file}")

# Example usage
if __name__ == "__main__":
    # Define two sets of corresponding points (Nx3 arrays)
    source_points = np.array([
        [1471086.788, 314418.818, 804.5449],
        [1472620.628, 314146.374, 945.6984],
        [1474165.094, 312911.379, 1101.5457],
        [1474886.470, 311129.312, 1117.4626],
        [1471741.985, 313178.215, 838.6653],
        [1472230.315, 311655.043, 831.1414],
        [1473255.315, 310563.556, 900.3956],
        [1472034.384, 310534.800, 769.5800]
    ])

    target_points = np.array([
        [-13396.303, -14644.817, 743.21],
        [-11861.662, -14913.39, 883.946],
        [-10313.968, -16144.082, 1039.308],
        [-9587.928, -17924.332, 1055.164],
        [-12737.766, -15883.932, 777.267],
        [-12245.299, -17405.726, 769.567],
        [-11217.529, -18494.435, 838.641],
        [-12438.326, -18526.413, 708.046]
    ])

    # Folder containing .las files
    folder_path = "."

    # Process all .las files in the folder
    process_las_files(folder_path, source_points, target_points)
