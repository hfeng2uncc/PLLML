# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:59:22 2025

@author: Jian Liu and Hongsong Feng
"""

"""
Main program for computing Rips complexes and analyzing vertex links.

The local Laplacian is obtained from the Laplacian of links.

Given a point cloud and a target point, this program performs the following:

1. Computes all pairwise distances between points in the cloud.
2. Generates Rips complexes at each distinct distance (distance threshold).
3. Maps the target point to its corresponding vertex index in the point cloud.
4. For each Rips complex:
   - Computes the link of the target vertex.
   - Analyzes the combinatorial Laplacian of the link (L_0 and L_1).
   - Outputs Betti numbers, spectral gaps, and the Laplacian matrices.
"""

from VR_complex import PointCloudRips
from link_calculator import LinkCalculator
import numpy as np

# from laplacian_calculator import LaplacianCalculator


def features_localpoint(eigenvalues_LP):

    eigenvalues_LP = np.array(eigenvalues_LP)
    tolerance = 1e-6
    positive_eigenvalues = np.array([e for e in eigenvalues_LP if e > tolerance])

    num_statistics = 9  # num of statistics features for each point in a point cloud
    # Calculate the mean, median, and standard deviation of the eigenvalues
    if len(positive_eigenvalues) > 0:
        features_point = [
            np.sum(positive_eigenvalues),
            np.mean(positive_eigenvalues),
            np.median(positive_eigenvalues),
            np.std(positive_eigenvalues),
            np.var(positive_eigenvalues),
            np.max(positive_eigenvalues),
            np.min(positive_eigenvalues),
            np.sum(np.power(positive_eigenvalues, 2)),
            np.shape(eigenvalues_LP[eigenvalues_LP < tolerance])[0],
        ]

    else:
        features_point = [0] * (num_statistics - 1)
        features_point.append(np.shape(eigenvalues_LP[eigenvalues_LP < tolerance])[0])

    return features_point


def _compute_statistical_features(features_points_list, stats):
    """
    Helper function to compute statistical features for a list of point features.

    Args:
        features_points_list: List of feature arrays for each filtration distance
        stats: List of statistical operations to perform

    Returns:
        Dictionary with statistical features computed across all filtration distances
    """
    # Initialize result dictionary with empty lists
    result_dict = {stat: [] for stat in stats}

    # Statistical operation mapping
    stat_operations = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "max": np.max,
        "min": np.min,
        "var": np.var,
    }

    # print("features_points_list",features_points_list)

    num_statistics = 9
    # Compute features for each filtration distance
    for features_at_distance in features_points_list:
        # print("features_at_distance",len(features_at_distance))
        if len(features_at_distance) == 0:
            # print("enter zero statistics....")
            # Handle empty case - extend with zeros
            for stat in stats:
                result_dict[stat].extend(
                    [0.0] * num_statistics
                )  # 9 features from features_eigenvalues
        else:

            # print("enter non-zero statistics....")
            # Convert to numpy array for vectorized operations
            features_array = np.array(features_at_distance)

            # Compute each statistical measure
            for stat in stats:
                operation = stat_operations[stat]
                computed_features = operation(features_array, axis=0)
                result_dict[stat].extend(computed_features)

    return result_dict


def features_pointcloud(PRO_xyz, LIG_xyz, filtration_dists, max_dim=2):
    """
    Main program:
    - Compute pairwise distances of point cloud
    - Generate Rips complexes up to 3-dimensional simplices
    - Compute link of target point
    - Analyze combinatorial Laplacians of link (L0, L1)
    """

    print("start computing features.......")

    stats_cloud = ["sum", "mean", "median", "std", "max", "min", "var"]  # 7 statistics
    num_filtration_dists = len(filtration_dists)

    pointcloud = np.concatenate([PRO_xyz, LIG_xyz], axis=0).astype(float)
    features_pointcloud_dict_PRO = {}
    features_pointcloud_dict_LIG = {}

    # print("len(pointcloud)",len(pointcloud))
    # Initialize point cloud
    pc = PointCloudRips(pointcloud)
    # Generate Rips complexes for all distances with maximum 3-dimensional simplices
    rips_dict = pc.generate_rips_dict_by_all_distances(
        filtration_dists, max_dim=max_dim
    )

    features_points_dict_Prot = {}
    features_points_dict_Lig = {}
    for dim in range(max_dim):
        features_points_dict_Prot[dim] = [[] for _ in range(num_filtration_dists)]
        features_points_dict_Lig[dim] = [[] for _ in range(num_filtration_dists)]

    print(rips_dict.keys())

    for id_filtration, (dist, rips) in enumerate(rips_dict.items()):
        print(f"\nDistance threshold: {dist:.4f}")
        simplices = [tuple(s) for s in rips["simplices"]]
        # Compute link
        link_calc = LinkCalculator(simplices)
        for target_point in PRO_xyz:
            # Find index of target point
            vertex_idx = pc.point_index(target_point)

            # Iterate over each distance threshold
            link_analysis = link_calc.get_link_laplacian_analysis(
                vertex_idx, max_dimension=max_dim
            )  # Link analysis max 2

            # Analyze L0 and L1 Laplacians of the link
            for dim, analysis in link_analysis["laplacian_analysis"].items():
                eigenvalues_LP = analysis["eigenvalues"]
                features_point = features_localpoint(eigenvalues_LP)
                features_points_dict_Prot[dim][id_filtration].append(features_point)

        for target_point in LIG_xyz:
            # Find index of target point
            vertex_idx = pc.point_index(target_point)

            # Iterate over each distance threshold
            link_analysis = link_calc.get_link_laplacian_analysis(
                vertex_idx, max_dimension=max_dim
            )  # Link analysis max 2

            # Analyze L0 and L1 Laplacians of the link
            for dim, analysis in link_analysis["laplacian_analysis"].items():
                eigenvalues_LP = analysis["eigenvalues"]
                features_point = features_localpoint(eigenvalues_LP)
                features_points_dict_Lig[dim][id_filtration].append(features_point)

    # Compute statistical features using the helper function

    for dim in range(max_dim):
        # print(dim,"PRO",features_points_dict_Prot[dim])
        features_pointcloud_dict_PRO[dim] = _compute_statistical_features(
            features_points_dict_Prot[dim], stats_cloud
        )
        features_pointcloud_dict_LIG[dim] = _compute_statistical_features(
            features_points_dict_Lig[dim], stats_cloud
        )

    return features_pointcloud_dict_PRO, features_pointcloud_dict_LIG
