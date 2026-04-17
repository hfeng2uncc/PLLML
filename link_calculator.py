# -*- coding: utf-8 -*-
"""
Link Calculator
Specialized class for computing the link of a vertex in a simplicial complex
"""

from typing import Set, List, Tuple, Any, Dict
from laplacian_calculator import LaplacianCalculator


class LinkCalculator:
    """Class for computing links in a simplicial complex"""

    def __init__(self, simplices: List[Tuple]):
        """
        Initialize the link calculator

        Args:
            simplices: List of simplices
        """
        self.simplices = set()
        for simplex in simplices:
            self.add_simplex(simplex)

    def add_simplex(self, simplex: Tuple):
        """Add a simplex and all of its faces"""
        self.simplices.add(simplex)

        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            if face:
                self.simplices.add(face)

    def contains_vertex(self, simplex: Tuple, vertex: Any) -> bool:
        """Check if a simplex contains a given vertex"""
        return vertex in simplex

    def get_closed_star(self, vertex: Any) -> Set[Tuple]:
        """
        Compute the closed star of vertex v (ClSt_K(v))

        ClSt_K(v) = {τ ∈ K | ∃σ ∈ K with v ∈ σ and τ ⊆ σ}
        """
        closed_star = set()

        # Find all simplices containing vertex v
        containing_simplices = set()
        for simplex in self.simplices:
            if self.contains_vertex(simplex, vertex):
                containing_simplices.add(simplex)

        # For each simplex containing v, add it and all of its faces
        for simplex in containing_simplices:
            closed_star.add(simplex)

            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1 :]
                if face:
                    closed_star.add(face)

        return closed_star

    def get_link(self, vertex: Any) -> Set[Tuple]:
        """
        Compute the link of vertex v (Lk_K(v))

        Lk_K(v) = {τ ∈ ClSt_K(v) | v ∉ τ}
        """
        closed_star = self.get_closed_star(vertex)

        link = set()
        for simplex in closed_star:
            if not self.contains_vertex(simplex, vertex):
                link.add(simplex)

        return link

    def get_link_laplacian_analysis(
        self, vertex: Any, max_dimension: int = 2
    ) -> Dict[str, Any]:
        """
        Compute the link of a vertex and analyze its Laplacian properties

        Args:
            vertex: Vertex to compute the link for
            max_dimension: Maximum dimension for Laplacian analysis

        Returns:
            Dictionary containing the link and Laplacian analysis
        """
        link = self.get_link(vertex)

        # Print link statistics for debugging
        # num_vertices = sum(1 for s in link if len(s) == 1)
        # num_edges = sum(1 for s in link if len(s) == 2)
        # num_triangles = sum(1 for s in link if len(s) == 3)
        # print(
        #     f"Vertex {vertex} link has {len(link)} simplices: {num_vertices} vertices, {num_edges} edges, {num_triangles} triangles"
        # )

        if not link:
            return {"vertex": vertex, "link": set(), "laplacian_analysis": {}}

        # Use LaplacianCalculator to analyze the link, with max_dimension
        link_calculator = LaplacianCalculator(list(link), max_dimension=max_dimension)
        laplacian_analysis = {}

        # Compute L0 and L1 (include even if empty)
        for k in range(max_dimension):
            analysis = link_calculator.analyze_laplacian_properties(k)
            laplacian_analysis[k] = analysis  # Do not filter out matrix_size=0

        return {
            "vertex": vertex,
            "link": link,
            "link_calculator": link_calculator,
            "laplacian_analysis": laplacian_analysis,
        }

    def __str__(self):
        return f"LinkCalculator({len(self.simplices)} simplices)"


def calculate_link_for_vertex(
    complex_simplices: List[Tuple], vertex: Any
) -> Set[Tuple]:
    """
    Compute the link of a given vertex in a simplicial complex

    Args:
        complex_simplices: List of simplices in the simplicial complex
        vertex: Vertex for which to compute the link

    Returns:
        Set of simplices in the link
    """
    calculator = LinkCalculator(complex_simplices)
    return calculator.get_link(vertex)


def analyze_link_laplacian(
    complex_simplices: List[Tuple], vertex: Any, max_dimension: int = 2
) -> Dict[str, Any]:
    """
    Compute the link of a vertex and analyze its combinatorial Laplacian properties

    Args:
        complex_simplices: List of simplices in the simplicial complex
        vertex: Vertex for which to compute the link
        max_dimension: Maximum dimension for Laplacian analysis

    Returns:
        Dictionary containing the link and Laplacian analysis
    """
    calculator = LinkCalculator(complex_simplices)
    return calculator.get_link_laplacian_analysis(vertex, max_dimension)
