# -*- coding: utf-8 -*-

import numpy as np
import itertools


class PointCloudRips:
    """
    PointCloudRips class computes pairwise distances of a point cloud
    and generates Rips (Vietoris-Rips) complexes keyed by distance thresholds.

    Attributes:
        points: List of points, each represented as a tuple or list.
        n: Number of points in the point cloud.
        distance_dic: Dictionary storing pairwise Euclidean distances.
        sorted_distances: Sorted list of all distinct pairwise distances.
    """

    def __init__(self, points):
        """
        Initialize the PointCloudRips object.

        Args:
            points: List of points (each a tuple or list)
        """
        self.points = [tuple(p) for p in points]
        self.n = len(points)
        self.distance_dic = self._compute_pairwise_distances()
        # self.sorted_distances = self._get_sorted_distances()

    def _compute_pairwise_distances(self):
        """
        Compute all pairwise Euclidean distances between points.

        Returns:
            dic: Dictionary with keys (i,j) -> distance between point i and j
        """
        dic = {}
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                dic[(i, j)] = np.linalg.norm(np.array(p1) - np.array(p2))
        return dic

    def _get_sorted_distances(self):
        """
        Get all distinct pairwise distances (i < j) and sort ascendingly.

        Returns:
            List of sorted distances
        """
        distances = [
            self.distance_dic[(i, j)]
            for i in range(self.n)
            for j in range(i + 1, self.n)
        ]
        return sorted(distances)

    def generate_rips_for_distance(self, max_distance, max_dim=3):
        """
        Generate a Rips complex for a single distance threshold.

        Args:
            max_distance: Distance threshold for including edges
            max_dim: Maximum simplex dimension to include (default 3)

        Returns:
            rips_complex: Dictionary with keys:
                'vertices': list of vertex indices
                'simplices': list of simplices (each a list of vertex indices)
                'dimension': highest simplex dimension included
        """
        rips_complex = {
            "vertices": list(range(self.n)),
            "simplices": [],
            "dimension": -1,
        }
        for dim in range(min(self.n - 1, max_dim) + 1):
            for simplex in itertools.combinations(range(self.n), dim + 1):
                distances = [
                    self.distance_dic[i, j]
                    for i, j in itertools.combinations(simplex, 2)
                ]
                if all(d <= max_distance for d in distances):
                    rips_complex["simplices"].append(list(simplex))
                    rips_complex["dimension"] = dim
        return rips_complex

    def generate_rips_dict_by_all_distances(self,filtration_dists, max_dim=3):
        """
        Generate a dictionary of Rips complexes keyed by each distinct distance.

        Args:
            max_dim: Maximum simplex dimension to include (default 3)

        Returns:
            rips_dict: Dictionary {distance: rips_complex}
        """
        rips_dict = {}
        # for d in self.sorted_distances:
        for d in filtration_dists:
            rips_dict[d] = self.generate_rips_for_distance(d, max_dim=max_dim)
        return rips_dict

    def point_index(self, point):
        """
        Given a point coordinate, return its index in the internal point list.

        Args:
            point: A tuple or list representing a point coordinate

        Returns:
            Index of the point if it exists in the cloud; None otherwise
        """
        
        point = tuple(point)
        if point in self.points:
            return self.points.index(point)
        else:
            raise ValueError(f"Point {point} not found in the list of points.")


# # =========================
# # Example usage
# # =========================
# if __name__ == "__main__":
#     """
#     This main program demonstrates how to:
#     1. Initialize a PointCloudRips object with a set of points.
#     2. Retrieve the index of each point in the internal point list.
#     3. Generate Rips complexes for all distinct pairwise distances.
#     4. Print the vertices and simplices of each Rips complex.
#     """

#     points = [(0,0), (2,1), (3,3), (1,2), (3,4)]
#     pc = PointCloudRips(points)

#     # Print point indices
#     print("Point indices:")
#     for p in points:
#         idx = pc.point_index(p)
#         print(f"  Point {p} corresponds to vertex index: {idx}")

#     # Generate Rips complexes for all distances (max_dim=3 by default)
#     rips_dict = pc.generate_rips_dict_by_all_distances(max_dim=3)
#     for d, rips in rips_dict.items():
#         print(f"\nDistance threshold: {d:.4f}")
#         print(f"  Vertices: {rips['vertices']}")
#         print(f"  Simplices: {rips['simplices']}")
