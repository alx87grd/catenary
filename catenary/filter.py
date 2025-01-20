import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def remove_ground_plane(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Remove the ground plane from points.

    Parameters:
    - points: numpy array of shape (n_points, 3) representing the point cloud.
    - distance_threshold: Maximum distance a point can be from the plane to be considered an inlier.
    - ransac_n: Number of points to sample for generating a plane model.
    - num_iterations: Number of iterations for RANSAC.

    Returns:
    - non_ground_points: numpy array of shape (n_non_ground_points, 3) representing the point cloud without the ground plane.
    """

    # Convert numpy array to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Apply RANSAC to segment the ground plane
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)

    # Extract non-ground points
    non_ground_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Convert back to numpy array
    non_ground_points = np.asarray(non_ground_cloud.points)

    return non_ground_points

def find_clusters_dbscan(points, eps=0.02, min_samples=10)
    """
    Use DBSCAN to cluster points.

    Parameters:
    - points: numpy array of shape (n_points, 3) representing the point cloud.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - labels: Array of cluster labels for each point.
    """
    # Perform DBSCAN Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    return labels

def plot_clusters_3d(points, labels) -> None:
    """
    Plot the point cloud with a different RGB color for each cluster using scatter3D.

    Parameters:
    - points: numpy array of shape (n_points, 3) representing the point cloud.
    - labels: Array of cluster labels for each point.
    """
    # Number of clusters in labels, ignoring noise if present.
    unique_labels = set(labels)
    colors = plt.cm.prism(np.linspace(0, 1, len(unique_labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with a different color
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xyz = points[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], marker='o')

    ax.set_title('DBSCAN Clustering')
    plt.show()
    plt.pause(10)

def is_horizontal_line(cluster_points) -> bool:
    """
    Check if the cluster points form a horizontal line.

    Parameters:
    - cluster_points: numpy array of shape (n_points, 3) representing the cluster points.

    Returns:
    - True if the cluster points form a horizontal line, False otherwise.
    """

    # Calculate the covariance matrix of the cluster points
    covariance_matrix = np.cov(cluster_points.T)

    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Identify the major eigenvalue and its corresponding eigenvector
    major_eigenvalue_index = np.argmax(eigenvalues)
    major_eigenvalue = eigenvalues[major_eigenvalue_index]
    major_eigenvector = eigenvectors[:, major_eigenvalue_index]

    # Check if the major eigenvalue is significantly larger than the other eigenvalues
    other_eigenvalues = np.delete(eigenvalues, major_eigenvalue_index)
    if major_eigenvalue < 100 * np.max(other_eigenvalues):  # Adjust the factor as needed
        return False

    # Calculate the angle between the major eigenvector and the horizontal plane (XY-plane)
    horizontal_component = np.linalg.norm(major_eigenvector[:2])
    vertical_component = np.abs(major_eigenvector[2])
    angle = np.arctan2(vertical_component, horizontal_component) * 180 / np.pi

    # Return True if the angle is less than or equal to 45 degrees
    return angle <= 45

def keep_line_clusters(points, labels):
    """
    Keep clusters that have the shape of a horizontal line and return merged lines' points.

    Parameters:
    - points: numpy array of shape (n_points, 3) representing the point cloud.
    - labels: Array of cluster labels for each point.

    Returns:
    - merged_line_points: numpy array of shape (n_merged_points, 3) representing the merged points of line-shaped clusters.
    """
    unique_labels = set(labels)
    line_clusters = []

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        cluster_points = points[labels == label]
        if cluster_points.shape[0] < 3:
            continue  # Skip clusters with less than 3 points

        if is_horizontal_line(cluster_points):
            line_clusters.append(cluster_points)

    if line_clusters:
        merged_line_points = np.vstack(line_clusters)
    else:
        merged_line_points = np.empty((0, points.shape[1]))

    return merged_line_points

def filter_cable_points(points):
    """
    Filter the cable points from the point cloud.

    Parameters:
    - points: numpy array of shape (n_points, 3) representing the point cloud.

    Returns:
    - merged_line_points: numpy array of shape (n_merged_points, 3) representing the merged points of line-shaped clusters.
    """

    # Remove NaNs if any
    points = points[~np.isnan(points).any(axis=1)]

    # Remove ground plane
    non_ground_points = remove_ground_plane(points, distance_threshold=5.0, ransac_n=5)
    #non_ground_points = points

    # Apply DBSCAN clustering
    labels = find_clusters_dbscan(non_ground_points, eps=1.0, min_samples=3)

    # For debuggign purpose
    if False:
        plot_clusters_3d(non_ground_points, labels)

    # Extract clusters that looks like line
    merged_line_points = keep_line_clusters(non_ground_points, labels)

    return merged_line_points
