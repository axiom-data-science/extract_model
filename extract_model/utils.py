#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities to help extract_model work better.
"""
from itertools import product
from typing import List, Mapping, NewType, Optional, Tuple

import dask
import numpy as np
import xarray as xr

from sklearn.neighbors import BallTree

from extract_model.grids.triangular_mesh import UnstructuredGridSubset
from extract_model.model_type import ModelType


BBoxType = NewType("BBoxType", Tuple[float, float, float, float])


def filter(
    ds: xr.Dataset,
    standard_names: List[str],
    keep_horizontal_coords: bool = True,
    keep_vertical_coords: bool = True,
    keep_coord_mask: bool = True,
    model_type: Optional[ModelType] = None,
):
    """Filter Dataset by variables

    ... but retain all necessary for decoding vertical coords.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset to select model output from.
    standard_names : list of strings
        Standard names of variables to keep in Dataset.
    keep_horizontal_coords : bool
        Optionally include all horizontal coordinate variables which can map to lon/lat. Defauls to
        True.
    keep_vertical_coords : bool
        Optionally include vertical coordinate variables describing ocean sigma coordinates.
        Defaults to True
    keep_coord_mask : bool
        Optionally include variables that provide masks of the coordinate features. Defaults to
        True.

    Returns
    -------
    xr.Dataset
        Dataset with variables from standard_names included as well as variables corresponding to
        formula_terms needed to decode vertical coordinates using `cf-xarray`.
    """
    to_merge = []
    if model_type is not None and not isinstance(model_type, ModelType):
        model_type = ModelType(model_type)

    model_type_guess = model_type or guess_model_type(ds)

    if model_type_guess == "FVCOM":
        to_merge.append(ds[UnstructuredGridSubset.FVCOM_COORDINATE_VARIABLES])
    elif model_type_guess == "SELFE":
        to_merge.append(ds[UnstructuredGridSubset.SELFE_COORDINATE_VARIABLES])

    if "angle" in ds.variables:
        to_merge.append(ds["angle"])

    if keep_vertical_coords:
        # Deal with vertical coord decoding

        # standard_names associated with vertical coordinates
        s_standard_names_list = [
            "ocean_s_coordinate_g1",
            "ocean_s_coordinate_g2",
            "ocean_sigma_coordinate",
        ]

        # want to find the vertical coord standard_names variables AND those with formula_terms
        # which should be identical but seems like a good check
        def formula_terms(value):
            """weed out Nones"""
            return value is not None

        def s_standard_names(value):
            """loop over s coord standard_names"""
            return value in s_standard_names_list

        # get a Dataset with these coordinates
        v_grid_ds = ds.filter_by_attrs(
            formula_terms=formula_terms, standard_name=s_standard_names
        )
        if len(v_grid_ds.variables) > 0:
            to_merge.append(v_grid_ds)

        # For the vertical related coords (e.g. for ROMS these will be `s_rho` and `s_w`
        # gather all formula term variable names to bring along
        formula_vars = []
        for coord in v_grid_ds.coords:
            # It's possible to have coordinates of the filtered variables which do not define
            # formula_terms, like in FVCOM.
            if "formula_terms" in v_grid_ds[coord].attrs:
                formula_vars.extend(list(v_grid_ds[coord].cf.formula_terms.values()))
        formula_vars = list(set(formula_vars))
        if len(formula_vars) > 0:
            to_merge.append(ds[formula_vars])

    if keep_horizontal_coords:
        # Get a ds for coordinates
        h_coords_standard_names = [
            "longitude",
            "latitude",
        ]
        h_grid_ds = ds.filter_by_attrs(
            standard_name=lambda v: v in h_coords_standard_names
        )
        if len(h_grid_ds.variables) > 0:
            to_merge.append(h_grid_ds)
        if model_type_guess == "FVCOM":
            if "x" in ds.variables and "y" in ds.variables:
                to_merge.append(ds[["x", "y"]])
            if "xc" in ds.variables and "yc" in ds.variables:
                to_merge.append(ds[["xc", "yc"]])
        elif model_type_guess == "SELFE":
            if "x" in ds.variables and "y" in ds.variables:
                to_merge.append(ds[["x", "y"]])

    if keep_horizontal_coords and keep_coord_mask:
        # Keep coordinate masks
        mask_ds = ds.filter_by_attrs(flag_meanings="land water")
        if len(mask_ds.variables) > 0:
            to_merge.append(mask_ds)

    # Also get a Dataset for all the requested variables
    def f_standard_names(value):
        """loop over standard_names list"""
        return value in standard_names

    to_merge.append(ds.filter_by_attrs(standard_name=f_standard_names))

    # Combine
    return xr.merge(to_merge)


def naive_subbox(ds: xr.Dataset, bbox: BBoxType, dask_array_chunks: bool = False):
    """Perform subsetting directly using dimension slicing.

    naive_subbox performs subsetting by means of directly slicing along the grid
    dimensions in lieu of using a mask combined with da.where. Slicing along the
    dimensions directly is useful when access data remotely where bandwidth is a
    significant concern. This approach ensures that requests for data will
    request the minimal set of data possible.

    Parameters
    ----------
    ds : xr.Dataset
        The gridded data to subset.
    bbox : BBoxType (tuple)
        The desired codomain.
    dask_array_chunks : bool
        Set to True to use dask array chunking for large grids that don't fit
        into memory.
    """
    model_type = guess_model_type(ds)
    lons = ds.cf[["longitude"]]
    lon_coords = sorted(set(lons.coords) - set(lons.dims))

    lats = ds.cf[["latitude"]]
    lat_coords = sorted(set(lats.coords) - set(lats.dims))

    lon_is_coordinate_variable = False
    lat_is_coordinate_variable = False
    # Check if lon is a coordinate variable
    if len(ds.cf.standard_names["longitude"]) == 1:
        varname = ds.cf.standard_names["longitude"][0]
        if ds[varname].dims == (varname,):
            lon_is_coordinate_variable = True
    if len(ds.cf.standard_names["latitude"]) == 1:
        varname = ds.cf.standard_names["latitude"][0]
        if ds[varname].dims == (varname,):
            lat_is_coordinate_variable = True

    if lon_is_coordinate_variable != lat_is_coordinate_variable:
        raise ValueError("Invalid grid detected")

    if lon_is_coordinate_variable:
        lon_data = ds.cf["longitude"][:]
        lat_data = ds.cf["latitude"][:]
        lon_mask = np.where((lon_data >= bbox[0]) & (lon_data <= bbox[2]))[0]
        lat_mask = np.where((lat_data >= bbox[1]) & (lat_data <= bbox[3]))[0]
        x0 = np.min(lon_mask)
        x1 = np.max(lon_mask)
        y0 = np.min(lat_mask)
        y1 = np.max(lat_mask)
        with dask.config.set(**{"array.slicing.split_large_chunks": dask_array_chunks}):
            return ds.cf.isel(longitude=slice(x0, x1), latitude=slice(y0, y1))

    grid_mappings = []

    for lon_coord in lon_coords:
        for lat_coord in lat_coords:
            if ds[lon_coord].dims == ds[lat_coord].dims:
                grid_mappings.append(
                    {"lon": lon_coord, "lat": lat_coord, "dims": ds[lon_coord].dims}
                )
                break
    slices = {}

    # Staggered grid case for ROMS
    roms_native_coords = ["eta", "xi"]
    roms_cell_coords = ["rho", "u", "v", "psi"]
    roms_coords = [
        f"{native}_{cell}"
        for cell, native in product(roms_cell_coords, roms_native_coords)
    ]
    if (
        model_type == "ROMS"
        and all([i in ds.coords for i in ["lon_rho", "lat_rho"]])
        and all([coord in ds.dims for coord in roms_coords])
    ):
        slices = _slice_grid_mapping(
            ds, "lon_rho", "lat_rho", ["eta_rho", "xi_rho"], bbox
        )
        eta_rho: slice = slices["eta_rho"]
        xi_rho: slice = slices["xi_rho"]
        slices["eta_u"] = eta_rho
        slices["xi_u"] = slice(xi_rho.start, xi_rho.stop - 1)

        slices["eta_v"] = slice(eta_rho.start, eta_rho.stop - 1)
        slices["xi_v"] = xi_rho

        slices["eta_psi"] = slice(eta_rho.start, eta_rho.stop - 1)
        slices["xi_psi"] = slice(xi_rho.start, xi_rho.stop - 1)

        with dask.config.set(**{"array.slicing.split_large_chunks": dask_array_chunks}):
            return ds.isel(**slices)

    for grid_mapping in grid_mappings:
        lon = grid_mapping["lon"]
        lat = grid_mapping["lat"]
        dims = grid_mapping["dims"]
        grid_slices = _slice_grid_mapping(ds, lon, lat, dims, bbox)
        slices.update(grid_slices)

    with dask.config.set(**{"array.slicing.split_large_chunks": dask_array_chunks}):
        return ds.isel(**slices)


def _slice_grid_mapping(
    ds: xr.Dataset, lon: str, lat: str, dims: List[str], bbox: BBoxType
) -> Mapping[str, slice]:
    slices = {}
    lon_data = ds[lon][:].to_numpy()
    lat_data = ds[lat][:].to_numpy()

    mask = (
        (lon_data >= bbox[0])
        & (lon_data <= bbox[2])
        & (lat_data >= bbox[1])
        & (lat_data <= bbox[3])
    )
    if not np.any(mask):
        raise ValueError("Bounding box does not intersect dataset domain")
    mask_i = np.where(mask)
    x0, y0 = np.min(mask_i[0]), np.min(mask_i[1])
    x1, y1 = np.max(mask_i[0]) + 1, np.max(mask_i[1]) + 1
    slices[dims[0]] = slice(x0, x1)
    slices[dims[1]] = slice(y0, y1)
    return slices


def sub_grid(
    ds: xr.Dataset,
    bbox: BBoxType,
    dask_array_chunks=True,
    model_type: Optional[ModelType] = None,
    naive: bool = False,
    preload: bool = False,
):
    """Subset Dataset grids.

    Preserves horizontal grid structure, which matters for ROMS.
    Like `sub_bbox`, this function takes in a bounding box to
    limit the horizontal domain size. But this function preserves
    the relative sizes of horizontal grids if more than 1. There
    will be grid values outside of bbox if grid is curvilinear since
    the axes then do not follow lon/lat directly.

    If there is only one horizontal grid, this simply calls `sub_bbox()`.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset to select model output from.
    bbox: list
        The bounding box for subsetting is defined as [min_lon, min_lat, max_lon, max_lat]
    dask_array_chunks: boolean, optional
        If True, avoids creating large chunks in slicing operation. If False, accept the large chunk
        and silence this warning. Comes up if Slicing is producing a large chunk.
    model_type : ModelType, optional
        Clients may explicitly specify the model type for the grid and data in `ds`. If this
        parameter isn't specified, it is guessed based on the metadata available.
    naive : bool
        Causes the subetting method to use naive_subset which has better performance over DAP.
    preload : bool
        If the subsetting algorithm supports it, the data will be preloaded
        before reindexed (only applicable to unstructured grids). Defaults to
        False.

    Returns
    -------
    Dataset that preserves original Dataset horizontal grid structure.

    """

    assertion = "Input should be `xarray` Dataset. For DataArray, try `sub_bbox()`."
    if not isinstance(ds, xr.Dataset):
        raise ValueError(assertion)

    if model_type is not None and not isinstance(model_type, ModelType):
        model_type = ModelType(model_type)

    model_type_guess = model_type or guess_model_type(ds)
    if model_type_guess == "FVCOM":
        subsetter = UnstructuredGridSubset()
        return subsetter.subset(ds=ds, bbox=bbox, grid_type="fvcom", preload=preload)
    if model_type_guess == "SELFE":
        subsetter = UnstructuredGridSubset()
        return subsetter.subset(ds=ds, bbox=bbox, grid_type="selfe", preload=preload)

    if naive:
        return naive_subbox(ds=ds, bbox=bbox, dask_array_chunks=dask_array_chunks)

    attrs = ds.attrs

    # return var names for longitude — may be more than 1
    lons = ds.cf[["longitude"]]
    lon_names = set(lons.coords) - set(lons.dims)

    # if longitude is also the name of the dimension
    if len(lon_names) == 0:
        lon_names = set(lons.coords)

    # check for ROMS special case
    if "lon_rho" in lon_names:
        # variables with 'lon_rho', just use first one
        varname = [v for v in ds.data_vars if "lon_rho" in ds[v].coords][0]

        # get xi_rho and eta_rho slice values
        # unfortunately the indices are reset when the array changes size
        # IF the dimensions are dims only and not coords
        if "xi_rho" not in ds.coords:
            subs = sub_bbox(ds[varname], bbox, other=-500, drop=False)

            # index
            i_xi_rho = int((subs != -500).sum(dim="xi_rho").argmax())
            xi_rho_bool = subs.isel(eta_rho=i_xi_rho) != -500
            if "T" in subs.cf.axes:
                xi_rho_bool = xi_rho_bool.cf.isel(T=0)
            if "Z" in subs.cf.axes:
                xi_rho_bool = xi_rho_bool.cf.isel(Z=0)
            xi_rho = np.arange(ds.lon_rho.shape[1])[xi_rho_bool]

            i_eta_rho = int((subs != -500).sum(dim="eta_rho").argmax())
            eta_rho_bool = subs.isel(xi_rho=i_eta_rho) != -500
            if "T" in subs.cf.axes:
                eta_rho_bool = eta_rho_bool.cf.isel(T=0)
            if "Z" in subs.cf.axes:
                eta_rho_bool = eta_rho_bool.cf.isel(Z=0)
            eta_rho = np.arange(ds.lon_rho.shape[0])[eta_rho_bool]

        else:  # 'xi_rho' in ds.coords
            # this works in this case because the dimensions as coords can
            # "remember" their original values
            subsetted = sub_bbox(ds[varname], bbox, drop=True)
            # get xi_rho and eta_rho slice values
            xi_rho, eta_rho = subsetted.xi_rho.values, subsetted.eta_rho.values

        # This first part is to keep the dimensions consistent across
        # the grids
        # then know xi_u, eta_v
        sel_dict = {"xi_rho": xi_rho, "eta_rho": eta_rho}
        if "xi_u" in ds.dims:
            sel_dict["xi_u"] = xi_rho[:-1]
        if "eta_v" in ds.dims:
            sel_dict["eta_v"] = eta_rho[:-1]
        if "eta_u" in ds.dims:
            sel_dict["eta_u"] = eta_rho
        if "xi_v" in ds.dims:
            sel_dict["xi_v"] = xi_rho
        if "eta_psi" in ds.dims:
            sel_dict["eta_psi"] = eta_rho[:-1]
        if "xi_psi" in ds.dims:
            sel_dict["xi_psi"] = xi_rho[:-1]
        # adjust dimensions of full dataset
        import dask

        with dask.config.set(**{"array.slicing.split_large_chunks": dask_array_chunks}):
            ds_new = ds.sel(sel_dict)

    elif len(lon_names) == 1:

        ds_new = sub_bbox(ds, bbox, drop=True)

    else:
        # raise exception
        print("Not prepared to deal with this situation.")

    ds_new.attrs = attrs

    return ds_new


def sub_bbox(da, bbox, other=xr.core.dtypes.NA, drop=True, dask_array_chunks=True):
    """Subset DataArray in space.

    Can also be used on a Dataset if there is only one horizontal grid.

    Parameters
    ----------
    da: DataArray
        Property to select model output from.
    bbox: list
        The bounding box for subsetting is defined as [min_lon, min_lat, max_lon, max_lat]
    other: int, float, optional
        Value to input in da where bbox is False. Either other or drop is used. By default is nan.
    drop: bool, optional
        This is passed onto xarray's `da.where()` function. If True, coordinates outside bbox
        are dropped from the DataArray, otherwise they are kept but masked/nan'ed.
    dask_array_chunks: boolean, optional
        If True, avoids creating large chunks in slicing operation. If False, accept the large chunk and silence this warning. Comes up if Slicing is producing a large chunk.

    Notes
    -----
    Not dealing with MOM6 output currently.
    """

    # this condition defines the region of interest
    box = (
        (bbox[0] < da.cf["longitude"])
        & (da.cf["longitude"] < bbox[2])
        & (bbox[1] < da.cf["latitude"])
        & (da.cf["latitude"] < bbox[3])
    ).compute()

    import dask

    with dask.config.set(**{"array.slicing.split_large_chunks": dask_array_chunks}):
        da_smaller = da.where(box, other=other, drop=drop)

    return da_smaller


def order(da):
    """Reorder var to typical dimensional ordering.

    Parameters
    ------
    da: DataArray
        Variable to operate on.

    Returns
    -------
    DataArray with dimensional order ['T', 'Z', 'Y', 'X'], or whatever subset of
    dimensions are present in da.

    Notes
    -----
    Does not consider previously-selected dimensions that are kept on as coordinates but
    cannot be transposed anymore. This is accomplished with `.reset_coords(drop=True)`.

    Examples
    --------
    >>> em.order(da)
    """

    return da.cf.transpose(
        *[
            dim
            for dim in ["T", "Z", "Y", "X"]
            if dim in da.reset_coords(drop=True).cf.axes
        ]
    )


def tree_query(
    lon_coords: xr.DataArray,
    lat_coords: xr.DataArray,
    lons_to_find: np.array,
    lats_to_find: np.array,
    k: int = 3,
) -> Tuple[np.array]:
    """Set up and query BallTree for k nearest points

    Uses haversine for the metric because we are dealing with lon/lat coordinates.

    Parameters
    ----------
    lon_coords : xr.DataArray
        Longitude coordinates of grid you are searching for nearest points on.
    lat_coords : xr.DataArray
        Latitude coordinates of grid you are searching for nearest points on.
    lons_to_find : np.array
        Longitudes of points you are searching for nearest grid points to.
    lats_to_find : np.array
        Latitudes of points you are searching for nearest grid points to.
    k : int, optional
        Number of nearest points to return, by default 3

    Returns
    -------
    Tuple[np.array]
        distances, (iys, ixs) 2D indices for coordinates

    Notes
    -----
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    """

    # create tree
    coords = [lon_coords, lat_coords]
    X = np.stack([np.ravel(c) for c in coords]).T
    tree = BallTree(np.deg2rad(X), metric="haversine")

    # set up coordinates we want to search for
    coords_to_find = [lons_to_find, lats_to_find]
    X_to_find = np.stack([np.ravel(c) for c in coords_to_find]).T

    # query tree
    distances, inds = tree.query(np.deg2rad(X_to_find), k=k)

    # convert flat indies to 2D indices
    iys, ixs = np.unravel_index(inds, lon_coords.shape)

    return distances, (iys, ixs)


def calc_barycentric(
    x: np.array, y: np.array, xs: np.array, ys: np.array
) -> xr.DataArray:
    """Calculate barycentric weights for npts

    Parameters
    ----------
    x
        npts x 1 vector of x locations, can be in lon or projection coordinates.
    y
        npts x 1 vector of y locations, can be in lat or projection coordinates.
    xs
        npts x 3 array of triangle x vertices with which to calculate the barycentric weights for each of npts
    ys
        npts x 3 array of triangle y vertices with which to calculate the barycentric weights for each of npts

    Returns
    -------
    xr.DataArray
        Lambda, npts x 3 containing for each of npts the 3 barycentric weights to use for interpolation.
    """
    # barycentric weights
    # npts x 1 (vectors)
    L1 = (
        (ys[:, 1] - ys[:, 2]) * (x[:] - xs[:, 2])
        + (xs[:, 2] - xs[:, 1]) * (y[:] - ys[:, 2])
    ) / (
        (ys[:, 1] - ys[:, 2]) * (xs[:, 0] - xs[:, 2])
        + (xs[:, 2] - xs[:, 1]) * (ys[:, 0] - ys[:, 2])
    )
    L2 = (
        (ys[:, 2] - ys[:, 0]) * (x[:] - xs[:, 2])
        + (xs[:, 0] - xs[:, 2]) * (y[:] - ys[:, 2])
    ) / (
        (ys[:, 1] - ys[:, 2]) * (xs[:, 0] - xs[:, 2])
        + (xs[:, 2] - xs[:, 1]) * (ys[:, 0] - ys[:, 2])
    )
    L3 = 1 - L1 - L2

    lam = xr.DataArray(dims=("npts", "triangle"), data=np.vstack((L1, L2, L3)).T)

    return lam


def interp_with_barycentric(da, ixs, iys, lam):
    vector = da.cf.isel(
        X=xr.DataArray(ixs, dims=("npts", "triangle")),
        Y=xr.DataArray(iys, dims=("npts", "triangle")),
    )
    with xr.set_options(keep_attrs=True):
        da = xr.dot(vector, lam, dim=("triangle"))

    # get z coordinates to go with interpolated output if not available
    if "vertical" in vector.cf.coords:
        zkey = vector.cf["vertical"].name

        # only need to interpolate z coordinates if they are not 1D
        if vector[zkey].ndim > 1:
            da_vert = xr.dot(vector[zkey], lam, dim=("triangle"))

            # add vertical coords into da
            da = da.assign_coords({zkey: da_vert})

    # add "X" axis to npts
    da["npts"] = ("npts", da.npts.values, {"axis": "X"})

    return da, vector.coords


def guess_model_type(ds: xr.Dataset) -> Optional[ModelType]:
    """Returns a guess as to which model produced the dataset."""
    selfe_dims = ["nele", "node"]
    for model_type in ModelType:
        if model_type in "".join([str(val) for val in ds.attrs.values()]):
            if model_type == "FVCOM" and not all(
                ["nv" in ds.variables, "node" in ds.dims]
            ):
                return None
            if model_type == "SELFE" and not all([i in ds.dims for i in selfe_dims]):
                return None
            return ModelType(model_type)
    return None
