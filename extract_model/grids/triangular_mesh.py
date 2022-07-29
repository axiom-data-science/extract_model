#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithms and utilties for triangular meshes."""
import typing

from typing import NewType, Tuple

import numpy as np
import xarray as xr

from numba import njit


# Literal isn't supported in Python 3.7
if typing.TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    GridType = Literal["fvcom", "selfe"]


BBOXType = NewType("BBOXType", Tuple[float, float, float, float])


@njit
def index_of_sorted(haystack: np.array, values: np.array) -> np.array:  # pragma: no cover
    """Return an array of indexes for each value in values found in haystack.

    This function uses binary search on haystack to find each value in values and returns an array
    of indices or -1 if an exact value is not identified. This function behaves similarly to
    np.searchsorted but will return -1 if there is no exact value.

    Parameters
    ----------
    haystack: np.ndarray
        A _sorted_ array of values from which each value in values array is matched to.
    values: np.ndarray
        An array of values to search for.

    Returns
    -------
    np.ndarray
        An array indices which such that
    """
    out = np.full_like(values, -1, dtype=np.int32)
    n = haystack.shape[0]

    for i, search_val in np.ndenumerate(values):
        left = 0
        right = n - 1
        if search_val < haystack[0] or search_val > haystack[-1]:
            out[i] = -1
            continue
        while left <= right:
            m = (left + right) // 2
            if haystack[m] < search_val:
                left = m + 1
            elif haystack[m] > search_val:
                right = m - 1
            else:
                out[i] = m
                break
    return out


class UnstructuredGridSubset:
    """A class for subsetting unstructured grids."""

    def __init__(self):
        """Initializes the subsetting class."""
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
        """Return the signed area of the vertices multiplied by 2.

        This function is used for fast checking if points reside inside a triangle.
        """
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

    def _get_intersecting_mask(
        self, x: np.ndarray, y: np.ndarray, element: np.ndarray, bbox
    ) -> np.ndarray:
        """Return a mask for the region subsetted by bbox."""
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

    def get_intersecting_mask(
        self, ds: xr.Dataset, bbox: BBOXType, grid_type: "GridType"
    ) -> np.ndarray:
        """Return a mask of only the valid triangles that intersect the bounding box."""
        # The triangulation indices
        # FVCOM uses the nv variable to indicate triangle connectivity. It is 1-indexed so in order
        # to make it work with anything but Fortran we subtract 1 to make it valid indexing.
        if grid_type == "fvcom":
            element = np.swapaxes(ds["nv"][:].to_numpy(), 0, 1) - 1
            # Longitude of vertices
            x = ds["lon"][:].to_numpy()
            # Latitude of Vertices
            y = ds["lat"][:].to_numpy()
            return self._get_intersecting_mask(x, y, element, bbox)
        raise ValueError(f"Unsupported grid type {grid_type}")

    def subset(
        self, ds: xr.Dataset, bbox: BBOXType, grid_type: "GridType"
    ) -> xr.Dataset:
        """Returns a subsetted dataset."""
        if grid_type == "fvcom":
            return self._subset_fvcom(ds, bbox)

    def _subset_fvcom(self, ds: xr.Dataset, bbox: BBOXType) -> xr.Dataset:
        """Return an xarray Dataset that will contain a subsetted version of the data.

        Parameters
        ----------
        ds : xr.Dataset
            An open FVCOM dataset.
        bbox : Tuple of four floats
            The axis-aligned bounding box containing (xmin, ymin, xmax, ymax).

        Returns
        -------
        xr.Dataset
            A dataset object containing an entirely subsetted and self-describing dataset conforming
            to FVCOM metadata.

        Notes
        -----
            The variables art1 and art2 are discarded because the area for the faces can't be
            trivially recomputed.
        """

        element = ds["nv"][:].to_numpy().T - 1
        mask = self.get_intersecting_mask(ds, bbox, "fvcom")
        # Get a sorted array of each node that is in our list of triangles to keep
        node_indices = np.unique(np.sort(element[mask].flatten()))
        special_vars = [
            "nv",
            "nbe",
            "ntsn",
            "nbsn",
            "ntve",
            "nbve",
            "art1",
            "art2",
        ]
        variables = {}
        for varname in ds.variables:
            if varname in special_vars:
                continue
            if len(ds[varname].dims) < 1:
                xvar = self._fvcom_copy_variable(ds, varname)
                variables[varname] = xvar
            elif ds[varname].dims[-1] == "nele":
                xvar = self._fvcom_subset_variable(ds, varname, mask)
                variables[varname] = xvar
            elif ds[varname].dims[-1] == "node":
                xvar = self._fvcom_reindex_variable(ds, varname, node_indices)
                variables[varname] = xvar
            else:
                xvar = self._fvcom_copy_variable(ds, varname)
                variables[varname] = xvar

        variables["nv"] = self._fvcom_recompute_nv(ds, node_indices, mask)
        variables["nbe"] = self._fvcom_recompute_nbe(ds, mask)
        variables["nbsn"] = self._fvcom_recompute_nbsn(ds, node_indices)
        variables["ntsn"] = self._fvcom_recompute_ntsn(ds, node_indices, variables)
        variables["nbve"] = self._fvcom_recompute_nbve(ds, node_indices, mask)
        variables["ntve"] = self._fvcom_recompute_ntve(ds, node_indices, variables)

        ds_ = xr.Dataset(variables, attrs=ds.attrs)
        return ds_

    def _fvcom_copy_variable(self, ds, varname) -> xr.DataArray:
        """Return an exact copy of the variable as a DataArray"""
        if len(ds[varname].dims) < 1:
            data = ds[varname].to_numpy()
        else:
            data = ds[varname][:]
        xvar = xr.DataArray(
            data=data,
            dims=ds[varname].dims,
            attrs=ds[varname].attrs,
        )
        return xvar

    def _fvcom_subset_variable(self, ds, varname, mask) -> xr.DataArray:
        """Return a variable on the face of elements subsetted with mask."""
        slices = []
        for dimname in ds[varname].dims:
            if dimname != "nele":
                slices.append(slice(None))
            elif dimname == "nele":
                slices.append(mask)
        slices = tuple(slices)
        data = ds[varname][slices]
        xvar = xr.DataArray(
            data=data,
            dims=ds[varname].dims,
            attrs=ds[varname].attrs,
        )
        return xvar

    def _fvcom_reindex_variable(self, ds, varname, node_indices) -> xr.DataArray:
        """Return a variable on a node that has been reindexed with new node indices."""
        slices = []
        for dimname in ds[varname].dims:
            if dimname != "node":
                slices.append(slice(None))
            elif dimname == "node":
                slices.append(node_indices)
        slices = tuple(slices)
        data = ds[varname][slices]
        xvar = xr.DataArray(
            data=data,
            dims=ds[varname].dims,
            attrs=ds[varname].attrs,
        )
        return xvar

    def _fvcom_recompute_nv(self, ds, node_indices, mask) -> xr.DataArray:
        """Return the recomputed surrounding elements variable.

        In FVCOM, nv is a variable containing the surrounding nodes (vertices) for a given element
        (triangle). This function computes a new nv variable after subsetting.
        """
        nv = ds["nv"][:].to_numpy().swapaxes(0, 1) - 1

        # Create a new element variable that has the triangle indices by a reverse mapping from the old triangle_ids to the new
        nv_ = index_of_sorted(node_indices, nv[mask])
        data = nv_.T + 1
        xvar = xr.DataArray(
            data=data,
            dims=("three", "nele"),
            attrs=ds["nv"].attrs,
        )
        return xvar

    def _fvcom_recompute_nbe(self, ds, mask) -> xr.DataArray:
        """Return the recomputed element surrounding elements variable.

        In FVCOM, the nbe variable contains the index of each surrounding element (triangle) to a
        given element (triangle). This function computes a new nbe variable after subsetting.
        """
        # A mapping from the new element index to the old index
        inv_element_lookup = np.where(mask)[0]

        # nbe is the index of the surrounding elements for an element
        nbe = ds["nbe"][:].to_numpy().swapaxes(0, 1) - 1

        # Surrounding elements, reindexed
        nbe_ = index_of_sorted(inv_element_lookup, nbe[mask])

        data = nbe_.swapaxes(0, 1) + 1
        xvar = xr.DataArray(
            data=data,
            dims=("three", "nele"),
            attrs=ds["nbe"].attrs,
        )
        return xvar

    def _fvcom_recompute_nbsn(self, ds, node_indices) -> xr.DataArray:
        """Return the recomputed indices of neighboring nodes (vertices) to a node (vertex).

        In FVCOM, the nbsn variable contains indices of each neighboring node (vertex) to a given
        node (vertex). This function recomputes the indices of the neighbors if they exist after
        subsetting.
        """
        # nbsn is the index of neighboring vertices to a vertex
        nbsn = ds["nbsn"][:].to_numpy().swapaxes(0, 1) - 1

        # Compute the index of new neighboring vertices to a new vertex
        nbsn_ = np.full(
            (node_indices.shape[0], nbsn.shape[1]), fill_value=-1, dtype=np.int32
        )

        for new_vert_id in range(node_indices.shape[0]):
            old_vert_id = node_indices[new_vert_id]
            neighboring_vertices = nbsn[old_vert_id]
            reindexed_neighboring_vertices = index_of_sorted(
                node_indices, neighboring_vertices
            )
            nbsn_[new_vert_id] = reindexed_neighboring_vertices
        data = nbsn_.swapaxes(0, 1) + 1
        xvar = xr.DataArray(
            data=data,
            dims=("maxnode", "node"),
            attrs=ds["nbsn"].attrs,
        )
        return xvar

    def _fvcom_recompute_ntsn(self, ds, node_indices, variables) -> xr.DataArray:
        """Return the recomputed count of neighboring nodes (vertices) to a given node (vertex).

        In FVCOM, the ntsn variable contains the number of neighboring nodes (vertices) for a given
        node (vertex). This function recomputes the number of neighbors after subsetting.
        """
        nbsn_ = variables["nbsn"][:].to_numpy().swapaxes(0, 1) - 1
        # ntsn is the number of neighboring vertices to a vertex
        # ntsn_ is the new number of neighboring vertices to a new vertex
        ntsn_ = np.zeros(node_indices.shape, dtype=np.int32)
        for i, neighbors in enumerate(nbsn_):
            ntsn_[i] = np.sum(neighbors >= 0)
        xvar = xr.DataArray(
            data=ntsn_,
            dims=("node",),
            attrs=ds["ntsn"].attrs,
        )
        return xvar

    def _fvcom_recompute_nbve(self, ds, node_indices, mask) -> xr.DataArray:
        """Return the recomputed indices of elements (triangles) neighboring a node (vertex).

        In FVCOM, the nbve variable contains the index of neighboring elements (triangles) to a
        given node (vertex). This function recomputes the indices of neighbors after subsetting.
        """
        # Recompute nbve
        # nbve is an array of indexes of elements neighboring a node (vertex)
        nbve = ds["nbve"][:].to_numpy().swapaxes(0, 1) - 1
        # These are the new vertices after subsetting
        neighbors_of_new_vertices = nbve[node_indices]
        # An array of old indices of the new elements after the mask
        new_elements = np.where(mask)[0]
        # Create an empty (n,maxelem) array to hold new indices
        nbve_ = np.full((node_indices.shape[0], nbve.shape[1]), -1, dtype=np.int32)
        for i in range(node_indices.shape[0]):
            # compute each set of new element indices using index function
            nbve_[i] = index_of_sorted(new_elements, neighbors_of_new_vertices[i])
        # Change the indexing strategy and dimension order back to FVCOM format
        data = nbve_.swapaxes(0, 1) + 1
        xvar = xr.DataArray(
            data=data,
            dims=("maxelem", "node"),
            attrs=ds["nbve"].attrs,
        )
        return xvar

    def _fvcom_recompute_ntve(self, ds, node_indices, variables) -> xr.DataArray:
        """Return the recomputed count of neighboring elements (triangles) to a given node (vertex).

        In FVCOM, the ntve variable contains the number of neighboring elements (triangles) to a
        given node (vertex). This function recomputes the neighbors after subsetting.
        """
        nbve_ = variables["nbve"][:].to_numpy().swapaxes(0, 1) - 1
        ntve_ = np.zeros(node_indices.shape, dtype=np.int32)
        for i, neighbors in enumerate(nbve_):
            ntve_[i] = np.sum(neighbors >= 0)
        xvar = xr.DataArray(
            data=ntve_,
            dims=("node",),
            attrs=ds["ntve"].attrs,
        )
        return xvar
