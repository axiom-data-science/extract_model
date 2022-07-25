#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithms and utilties for triangular meshes."""
import numpy as np


class UnstructuredGridSubset:
    """A class for subsetting unstructured grids."""

    def __init__(self):
        pass

    def _compute_barycentrics_for_triangles(
        self, x: np.ndarray, y: np.ndarray, element: np.ndarray, point: np.ndarray
    ):
        """Return an array containing the barycentric coordinates of the point for each triangle.

        Parameters
        ----------
        x : np.ndarray
            The array of x-coordinates.
        y : np.ndarray
            The array of y-coordinates.
        element : np.ndarray of int, shape (nsimplex, 3)
            An array of indices of the points forming the simplices in the triangulation.
        point : np.ndarray shape (2,)
            A single point in R2 (x, y)

        Returns
        -------
        np.ndarray of float, shape (nsimplex, 3)
            The array returns the barycentric coordinates of the point relative to each triangle in
            element, such that the coordinates in L satisfy the following:

            p_0 = L_0 * x_0 + L_1 * x_1 + L_2 * x_2
            p_1 = L_0 * y_1 + L_1 * y_1 + L_2 * y_2

        Notes
        -----
            Normally algorithms for computing the barycentric coordinates compute for many points
            with a single triangle. This algorithm computes the barycentric coordinates for a single
            point, for many triangles.
        """
        # pylint: disable=invalid-name
        # Get an array (n, 2, 3). Outer dim is triangle id, 2nd dim is x or y, last dim is vertex of
        # triangle
        triangles = np.array([x[element], y[element]])
        # point is an array of shape (2,) which is simply the position vector of the point
        p = point
        # T is a (n, 2, 2) array
        # i is the ith triangle
        # j = 0 -> x coordinates
        # j = 1 -> y coordinates
        # k goes from 0 .. 1
        # Tⁱʲₖ = tʲⁱₖ - tʲⁱ₂
        T = np.empty((element.shape[0], 2, 2), dtype=x.dtype)
        # This loop is equivalent to
        # for i in range(element.shape[0]):
        #     for j in range(2):
        #         for k in range(2):
        #             T[i, j, k] = triangles[j, i, k] - triangles[j, i, 2]
        for i in range(element.shape[0]):
            # Use an empty inner dimension to ensure subtraction works as intended
            # Tⁱʲₖ = tʲⁱₖ - tʲⁱ₂
            T[i] = triangles[:, i, :2] - triangles[:, i, 2, None]
        # Create empty array to hold inverse T
        T_ = np.empty_like(T)
        # T_ = T⁻¹
        for i in range(element.shape[0]):
            T_[i] = np.linalg.inv(T[i])
        # Pⁱʲ = pʲ - tʲⁱ₂
        P = p - triangles[:, :, 2].T
        L = np.zeros((element.shape[0], 3), dtype=x.dtype)
        # Lⁱₖ = TⁱₖʲPⁱʲ
        L[:, :2] = np.einsum("ikj,ij->ik", T_, P)
        L[:, 2] = 1 - L[:, 0] - L[:, 1]
        return L

    def _mask_triangles_with_no_points_in_box(
        self, x: np.ndarray, y: np.ndarray, element: np.ndarray, bbox, mask: np.ndarray
    ):
        """Mask each triangle that does not contain at least one point in the bbox.

        Parameters
        ----------
        x : np.ndarray
            The array of x-coordinates.
        y : np.ndarray
            The array of y-coordinates.
        element : np.ndarray of int, shape (nsimplex, ndim+1)
            An array of indices of the points forming the simplices in the triangulation.
        bbox : array_like
            An array of the bounding box coordinates ordered: [x-min, y-min, x-max, y-max]
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.

        Returns
        -------
        tuple of np.ndarrays
            The first element of the tuple is an array of indices of the triangles that are masked
            by this function. The second element of the tuple is the mask array.
        """
        # Next identify which triangles have at least one point in the box
        x_vertices = x[element[mask]]
        y_vertices = y[element[mask]]
        intersecting_x = (bbox[0] <= x_vertices) & (x_vertices <= bbox[2])
        intersecting_y = (bbox[1] <= y_vertices) & (y_vertices <= bbox[3])
        intersecting_element = intersecting_x & intersecting_y
        intersecting_mask = np.any(intersecting_element, axis=1)
        submask = np.zeros_like(mask)
        submask = np.where(mask)[0][~intersecting_mask]
        mask[submask] = False
        return submask, mask

    def _mask_disjoint_triangles(
        self, x: np.ndarray, y: np.ndarray, element: np.ndarray, bbox, mask: np.ndarray
    ) -> np.ndarray:
        """Update the mask and set the mask value for all explicitly disjoint triangle to False

        Parameters
        ----------
        x : np.ndarray
            The array of x-coordinates.
        y : np.ndarray
            The array of y-coordinates.
        element : np.ndarray of int, shape (nsimplex, ndim+1)
            An array of indices of the points forming the simplices in the triangulation.
        bbox : array_like
            An array of the bounding box coordinates ordered: [x-min, y-min, x-max, y-max]
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.

        Returns
        -------
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.

        Notes
        -----
            In ℝ², a triangle cannot intersect an axis-aligned box if the axis-aligned bounding box
            for the triangle does not intersect the axis-aligned box.
        """
        # Get the vertices of each triangle as two arrays, one for x coordinates and one for y
        # coordinates
        element_x = x[element]
        element_y = y[element]
        # Compute the bounding boxes of each triangle2
        element_xmin = np.min(element_x, axis=1)
        element_xmax = np.max(element_x, axis=1)
        element_ymin = np.min(element_y, axis=1)
        element_ymax = np.max(element_y, axis=1)

        # Update the mask to filter triangles that are wholly disjoint: the bounding box of each
        # triangle lies completely outside the AABB.
        # Disable formatting to preseve readability of the masking expression
        # fmt: off
        mask[:] = (~((element_xmax < bbox[0]) | (element_xmin > bbox[2]))) &\
                  (~((element_ymax < bbox[1]) | (element_ymin > bbox[3])))
        # fmt: on
        return mask

    def _unmask_triangles_with_bbox_vertices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        element: np.ndarray,
        bbox,
        mask: np.ndarray,
        submask: np.ndarray,
    ):
        """Unmask triangles that have any of the BBOX vertices inside.

        Parameters
        ----------
        x : np.ndarray
            The array of x-coordinates.
        y : np.ndarray
            The array of y-coordinates.
        element : np.ndarray of int, shape (nsimplex, ndim+1)
            An array of indices of the points forming the simplices in the triangulation.
        bbox : array_like
            An array of the bounding box coordinates ordered: [x-min, y-min, x-max, y-max]
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.
        submask: np.ndarray of int
            The array triangle indices for which the vertices of the triangle did not reside in the
            BBOX

        Returns
        -------
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.
        """
        # pylint: disable=invalid-name
        # Next identify which triangles have any of the BBOX vertices in them
        points = np.array(
            [
                [bbox[0], bbox[1]],  # LL
                [bbox[0], bbox[3]],  # UL
                [bbox[2], bbox[3]],  # UR
                [bbox[2], bbox[1]],  # LR
            ]
        )

        # These are the triangles that did not have at least one vertex in the BBOX
        questionable_triangles = element[submask]

        for point in points:
            # Compute the barycentric coordinates for each triangle/point combination
            L = self._compute_barycentrics_for_triangles(
                x, y, questionable_triangles, point
            )
            # If all 3 barycentric coordinates have positive values, then the point resides in the
            # triangle.
            barycentric_intersection = np.sum(L > 0, axis=1) == 3
            # Unmask any triangle that has a BBOX point inside it.
            mask[submask[np.where(barycentric_intersection)]] = True
        return mask

    def _unmask_triangles_with_intersecting_edges(
        self,
        x: np.ndarray,
        y: np.ndarray,
        element: np.ndarray,
        bbox,
        mask: np.ndarray,
        submask: np.ndarray,
    ):
        """Unmask any triangle in the submask that has an edge that intersects with the BBOX.

        Parameters
        ----------
        x : np.ndarray
            The array of x-coordinates.
        y : np.ndarray
            The array of y-coordinates.
        element : np.ndarray of int, shape (nsimplex, ndim+1)
            An array of indices of the points forming the simplices in the triangulation.
        bbox : array_like
            An array of the bounding box coordinates ordered: [x-min, y-min, x-max, y-max]
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.
        submask: np.ndarray of int
            The array triangle indices for which the vertices of the triangle did not reside in the
            BBOX

        Returns
        -------
        mask : np.ndarray of bool
            The array which acts as a mask indicating which triangles intersect the bounding box.
        """
        edge_indices = [
            (0, 1),
            (0, 2),
            (1, 2),
        ]
        points = np.array(
            [
                [bbox[0], bbox[1]],  # LL
                [bbox[0], bbox[3]],  # UL
                [bbox[2], bbox[3]],  # UR
                [bbox[2], bbox[1]],  # LR
            ]
        )

        segments = np.array(
            [
                [points[0], points[1]],
                [points[1], points[2]],
                [points[2], points[3]],
                [points[0], points[3]],
            ]
        )

        def triangle_intersects_bbox(x_i, y_i, segments):
            # pylint: disable=invalid-name
            # Check each segment against each edge of the triangle, if any intersect immediately
            # return True, return False only if we checked everything.
            for segment in segments:
                a, b = segment
                for edge_i in edge_indices:
                    intersects = self._segment_intersects_segment(
                        a,
                        b,
                        [x_i[edge_i[0]], y_i[edge_i[0]]],
                        [x_i[edge_i[1]], y_i[edge_i[1]]],
                    )
                    if intersects:
                        return True
            return False

        for i, tri_i in enumerate(element[submask]):
            # x coordinates of the triangle in the submask
            x_i = x[tri_i]
            # y coordinates of the triangle in the submask
            y_i = y[tri_i]

            # If the triangle is already valid, we don't need to check it again
            if mask[submask[i]]:
                continue
            intersects = triangle_intersects_bbox(x_i, y_i, segments)
            if intersects:
                mask[submask[i]] = True

        return mask

    def _double_area_signed(self, a, b, c):
        # pylint: disable=invalid-name
        # The area of a triangle is related by
        #       | x₁ y₁ 1 |
        # 2 A = | x₂ y₂ 1 |
        #       | x₃ y₃ 1 |
        return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])

    def _segment_intersects_segment(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ):
        """Return True if the segments defined by points a, and b intersect the segment c, d.

        Parameters
        ----------
        a : np.ndarray of shape (2,)
            The first point of the first segment.
        b : np.ndarray of shape (2,)
            The second point of the first segment.
        c : np.ndarray of shape (2,)
            The first point of the second segment.
        d : np.ndarray of shape (2,)
            The second point of the second segment.

        Returns
        -------
        bool
            True if the segments intersect, False otherwise.
        """
        # pylint: disable=invalid-name
        a1 = self._double_area_signed(a, b, d)
        a2 = self._double_area_signed(a, b, c)
        if (a1 * a2) < 0.0:
            a3 = self._double_area_signed(c, d, a)
            a4 = a3 + a2 - a1

            if (a3 * a4) < 0.0:
                return True
        return False

    def get_intersecting_mask(self, nc, bbox) -> np.ndarray:
        """Return a mask of only the valid triangles that intersect the bounding box."""
        # The triangulation indices
        # FVCOM uses the nv variable to indicate triangle connectivity. It is 1-indexed so in order
        # to make it work with anything but Fortran we subtract 1 to make it valid indexing.
        element = np.swapaxes(nc["nv"][:], 0, 1) - 1
        # Longitude of vertices
        x = nc["lon"][:]
        # Latitude of Vertices
        y = nc["lat"][:]
        # The mask that will represent only valid elements
        mask = np.ones((element.shape[0]), dtype=np.bool)

        self._mask_disjoint_triangles(x, y, element, bbox, mask)
        submask, _ = self._mask_triangles_with_no_points_in_box(
            x, y, element, bbox, mask
        )
        self._unmask_triangles_with_bbox_vertices(x, y, element, bbox, mask, submask)
        self._unmask_triangles_with_intersecting_edges(
            x, y, element, bbox, mask, submask
        )
        return mask
