# -*- coding: utf-8 -*-
"""
Combinatorial Laplacian Calculator
Dedicated to computing the combinatorial Laplacian of a simplicial complex
Based on correct boundary matrix construction
"""

from typing import List, Tuple, Dict, Any

import numpy as np


class Matrix:
    """Wrapper around numpy array for Laplacian computations"""

    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.rows, self.cols = self.data.shape if self.data.size > 0 else (0, 0)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def shape(self):
        return self.data.shape

    def transpose(self):
        return Matrix(self.data.T)

    def __matmul__(self, other):
        return Matrix(self.data @ other.data)

    def __add__(self, other):
        return Matrix(self.data + other.data)

    def eigenvalues(self):
        """
        Return all eigenvalues sorted ascending,
        round to 9 significant digits and truncate tiny values to 0
        """
        if self.rows == 0 or self.cols == 0:
            return []

        eigvals = np.linalg.eigvalsh(self.data)

        # Round to 9 significant digits
        cleaned = [float(f"{lam:.9g}") for lam in eigvals]

        # Threshold tiny values to 0
        tol = 1e-12
        cleaned = [0.0 if abs(lam) < tol else lam for lam in cleaned]

        return sorted(cleaned)

    def smallest_positive_eigenvalue(self):
        """Return the smallest positive eigenvalue (λ > 0)"""
        eigvals = self.eigenvalues()
        for lam in eigvals:
            if lam > 0:
                return lam
        return None

    def rank(self):
        """Compute matrix rank"""
        return int(np.linalg.matrix_rank(self.data, tol=1e-10))

    def __str__(self):
        """Nice display, suppress scientific notation"""
        if self.rows == 0 or self.cols == 0:
            return "Matrix([])"
        with np.printoptions(precision=9, suppress=True):
            return f"Matrix({self.data})"


class LaplacianCalculator:
    """Class for computing combinatorial Laplacians"""

    def __init__(self, simplices: List[Tuple], max_dimension: int = 2):
        """
        Initialize the Laplacian calculator

        Args:
            simplices: List of simplices
            max_dimension: Maximum dimension to consider (default=2, i.e., only 0,1,2-simplices)
        """
        self.max_dimension = max_dimension
        self.simplices = set()
        for simplex in simplices:
            self.add_simplex(simplex)

        # Build ordered simplices (only up to max_dimension)
        self.ordered_simplices = self._build_ordered_simplices()
        self.dim_indices = self._compute_dim_indices()

    def add_simplex(self, simplex: Tuple):
        """Add a simplex and all its faces (up to max_dimension)"""
        simplex_dim = len(simplex) - 1
        if simplex_dim <= self.max_dimension:
            self.simplices.add(simplex)

            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1 :]
                if face:
                    face_dim = len(face) - 1
                    if face_dim <= self.max_dimension:
                        self.simplices.add(face)

    def _build_ordered_simplices(self) -> List[Tuple]:
        """Build an ordered list of simplices by dimension (only up to max_dimension)"""
        simplices_by_dim = {}
        for simplex in self.simplices:
            dim = len(simplex) - 1
            if dim <= self.max_dimension:
                if dim not in simplices_by_dim:
                    simplices_by_dim[dim] = []
                simplices_by_dim[dim].append(simplex)

        # Sort by dimension, then lex order inside each dimension
        ordered = []
        for dim in sorted(simplices_by_dim.keys()):
            ordered.extend(sorted(simplices_by_dim[dim]))

        return ordered

    def _compute_dim_indices(self) -> List[int]:
        # Return b[0]=0, b[1]=#0-simplices, b[2]=#0+#1, ...
        simplices_by_dim = self.get_simplices_by_dimension()
        max_dim = min(
            self.max_dimension, max(simplices_by_dim.keys()) if simplices_by_dim else 0
        )
        b = [0]
        cum = 0
        for dim in range(max_dim + 1):
            count = len([s for s in self.ordered_simplices if len(s) - 1 == dim])
            cum += count
            b.append(cum)
        return b

    def get_simplices_by_dimension(self) -> Dict[int, List[Tuple]]:
        """Group simplices by dimension (only up to max_dimension)"""
        simplices_by_dim = {}
        for simplex in self.simplices:
            dim = len(simplex) - 1
            if dim <= self.max_dimension:
                if dim not in simplices_by_dim:
                    simplices_by_dim[dim] = []
                simplices_by_dim[dim].append(simplex)
        return simplices_by_dim

    def compute_laplacian_matrices(self) -> List[Matrix]:
        """
        Compute combinatorial Laplacians for dimensions 0 and 1.
        Internally constructs boundary matrices.
        Returns [L_0, L_1].
        """
        # --- Prepare ordered_simplices and simplex_index ---
        ordered = [tuple(s) for s in getattr(self, "ordered_simplices", [])]
        num_simplices = len(ordered)
        if num_simplices == 0:
            return [Matrix([]), Matrix([])]

        # Map simplex -> global index
        simplex_index = {ordered[i]: i for i in range(num_simplices)}
        self.simplex_index = simplex_index  # save for later

        # --- Dimension indices ---
        if hasattr(self, "dim_indices") and isinstance(self.dim_indices, (list, tuple)):
            b = list(self.dim_indices)
        else:
            b = list(self._compute_dim_indices())

        if not b:
            b = [0, 0, 0, 0]
        elif len(b) < 4:
            b = b + [b[-1]] * (4 - len(b))

        # --- Build full boundary matrix ---
        boundary_full = [
            [0 for _ in range(num_simplices)] for _ in range(num_simplices)
        ]
        for i, simplex in enumerate(ordered):
            if len(simplex) <= 1:
                continue
            for j in range(len(simplex)):
                face = tuple(simplex[:j] + simplex[j + 1 :])
                face_idx = simplex_index.get(face)
                if face_idx is not None:
                    sign = 1 if (j % 2 == 0) else -1
                    boundary_full[i][face_idx] = sign

        # --- Extract submatrices ---
        lap_0_size = b[1] - b[0]
        if lap_0_size > 0:
            lap_0 = Matrix([[0 for _ in range(lap_0_size)] for _ in range(lap_0_size)])
        else:
            lap_0 = Matrix([])

        if b[2] > b[1]:
            boundary_1 = Matrix(
                [
                    [boundary_full[i][j] for j in range(b[0], b[1])]
                    for i in range(b[1], b[2])
                ]
            )
        else:
            boundary_1 = Matrix([])

        if b[3] > b[2]:
            boundary_2 = Matrix(
                [
                    [boundary_full[i][j] for j in range(b[1], b[2])]
                    for i in range(b[2], b[3])
                ]
            )
        else:
            boundary_2 = Matrix([])

        # --- Compute Laplacians ---
        if boundary_1.rows > 0 and boundary_1.cols > 0:
            laplacian_0 = lap_0 + (boundary_1.transpose() @ boundary_1)
        else:
            laplacian_0 = lap_0

        if boundary_1.rows == 0 or boundary_1.cols == 0:
            laplacian_1 = Matrix([])
        else:
            first_term = boundary_1 @ boundary_1.transpose()
            if boundary_2.rows > 0 and boundary_2.cols > 0:
                second_term = boundary_2.transpose() @ boundary_2
                laplacian_1 = first_term + second_term
            else:
                laplacian_1 = first_term

        return [laplacian_0, laplacian_1]

    def get_combinatorial_laplacian(self, k: int) -> Matrix:
        """
        Compute k-th combinatorial Laplacian (only supports k=0,1)
        """
        if k not in [0, 1]:
            return Matrix([])

        laplacian_matrices = self.compute_laplacian_matrices()

        if k == 0 and len(laplacian_matrices) > 0:
            return laplacian_matrices[0]
        elif k == 1 and len(laplacian_matrices) > 1:
            return laplacian_matrices[1]
        else:
            return Matrix([])

    def get_betti_number(self, k: int) -> int:
        """Compute k-th Betti number (# zero eigenvalues)"""
        L_k = self.get_combinatorial_laplacian(k)
        if L_k.rows == 0 or L_k.cols == 0:
            return 0

        eigenvalues = L_k.eigenvalues()
        tolerance = 1e-10
        zero_eigenvalues = sum(1 for e in eigenvalues if abs(e) < tolerance)

        return int(zero_eigenvalues)

    def get_spectral_gap(self, k: int) -> float:
        """Compute spectral gap (smallest positive eigenvalue)"""
        L_k = self.get_combinatorial_laplacian(k)
        if L_k.rows == 0 or L_k.cols == 0:
            return 0.0

        eigenvalues = L_k.eigenvalues()
        positive_eigenvalues = [e for e in eigenvalues if e > 1e-10]

        if len(positive_eigenvalues) == 0:
            return 0.0

        return float(min(positive_eigenvalues))

    def analyze_laplacian_properties(self, k: int) -> Dict[str, Any]:
        """Analyze Laplacian properties (only supports k=0,1)"""
        if k not in [0, 1]:
            return {
                "dimension": k,
                "matrix_size": 0,
                "betti_number": 0,
                "spectral_gap": 0.0,
                "eigenvalues": [],
                "rank": 0,
            }

        L_k = self.get_combinatorial_laplacian(k)

        if L_k.rows == 0 or L_k.cols == 0:
            return {
                "dimension": k,
                "matrix_size": 0,
                "betti_number": 0,
                "spectral_gap": 0.0,
                "eigenvalues": [],
                "rank": 0,
            }

        eigenvalues = L_k.eigenvalues()
        real_eigenvalues = [complex(e).real for e in eigenvalues]
        sorted_eigenvalues = sorted(real_eigenvalues)

        # tolerance = 1e-10
        # zero_count = sum(1 for e in sorted_eigenvalues if abs(e) < tolerance)

        # positive_eigenvalues = [e for e in sorted_eigenvalues if e > tolerance]
        # spectral_gap = float(min(positive_eigenvalues)) if positive_eigenvalues else 0.0

        return {
            "dimension": k,
            "matrix_size": L_k.rows,
            # "betti_number": int(zero_count),
            # "spectral_gap": spectral_gap,
            "eigenvalues": sorted_eigenvalues,
            "rank": L_k.rank(),
        }

    def __str__(self):
        return f"LaplacianCalculator({len(self.simplices)} simplices)"


# Example test code (commented out)
# def main():
#     simplices = [
#        (1, 2), (2,), (0, 3), (0, 2, 3), (1, 2, 3), (2, 3), (1,), (0, 2), (3,), (0,), (1, 3)
#     ]
#     lap_calc = LaplacianCalculator(simplices, max_dimension=2)
#     L0 = lap_calc.get_combinatorial_laplacian(0)
#     L1 = lap_calc.get_combinatorial_laplacian(1)
#     print("L0 (vertices Laplacian):")
#     print(L0.data)
#     print("\nL1 (edges Laplacian):")
#     print(L1.data)
#     print("\nL0 properties:", lap_calc.analyze_laplacian_properties(0))
#     print("\nL1 properties:", lap_calc.analyze_laplacian_properties(1))
#
# if __name__ == "__main__":
#     main()
