"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""

import warnings

from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
import xoak  # noqa: F401

from dask.delayed import Delayed
from xarray import DataArray, Dataset
from .utils import tree_query, calc_barycentric, interp_with_barycentric, order


try:
    import xesmf as xe

    XESMF_AVAILABLE = True
except ImportError:  # pragma: no cover
    XESMF_AVAILABLE = False  # pragma: no cover


# try:
# except ImportError:
#     warnings.warn(
#         "cartopy is not installed, so `sel2d` and `argsel2d` will not run.",
#         ImportWarning,
#     )


def interp_multi_dim(
    da: DataArray,
    da_out: Optional[DataArray] = None,
    # T: Optional[Union[str, list]] = None,
    # Z: Optional[Union[str, list]] = None,
    # iT: Optional[Union[int, list]] = None,
    # iZ: Optional[Union[int, list]] = None,
    extrap_method: Optional[str] = None,
    locstream: bool = False,
    weights=None,
):
    """Interpolate input DataArray to output DataArray using xESMF.

    Parameters
    ----------
    da: xarray.DataArray
        Input DataArray to interpolate.
    da_out: xarray.DataArray
        Output DataArray to interpolate to.
    T: datetime-like string, list of datetime-like strings, optional
    Z: int, float, list, optional
    iT: int or list of ints, optional
    iZ: int or list of ints, optional
    extrap_method: str, optional
    locstream: boolean, optional

    Returns
    -------
    DataArray of interpolated and/or selected values from da and array of interpolation weights.
    """

    if not XESMF_AVAILABLE:
        raise ModuleNotFoundError(  # pragma: no cover
            "xESMF is not available so horizontal interpolation in 2D cannot be performed."
        )
    
    da = da.chunk({"eta_rho": -1, "xi_rho": -1})

    # set up regridder, use weights if available
    regridder = xe.Regridder(
        da,
        da_out,
        "bilinear",
        extrap_method=extrap_method,
        locstream_out=locstream,
        ignore_degenerate=True,
        weights=weights,
    )

    # do regridding
    da_int = regridder(da, keep_attrs=True)
    if weights is None:
        weights = regridder.weights

    # get z coordinates to go with interpolated output if not available
    if "vertical" in da.cf.coords:
        zkey = da.cf["vertical"].name

        # only need to interpolate z coordinates if they are not 1D
        if da[zkey].ndim > 1:
            zint = regridder(da[zkey], keep_attrs=True)

            # add coords
            da_int = da_int.assign_coords({zkey: zint})

    return da_int, weights


def make_output_ds(longitude, latitude, locstream=False):
    """
    Given desired interpolated longitude and latitude, return points as Dataset.
    """

    # Grid of lat/lon to interpolate to with desired ending attributes
    if latitude.ndim == 1:
        if locstream:
            # for locstream need a single dimension (in this case called "loc")
            ds_out = xr.Dataset(
                {
                    "lat": (
                        ["loc"],
                        latitude,
                        dict(
                            axis="Y",
                            units="degrees_north",
                            standard_name="latitude",
                        ),
                    ),
                    "lon": (
                        ["loc"],
                        longitude,
                        dict(
                            axis="X",
                            units="degrees_east",
                            standard_name="longitude",
                        ),
                    ),
                }
            )

        else:
            ds_out = xr.Dataset(
                {
                    "lat": (
                        ["lat"],
                        latitude,
                        dict(
                            axis="Y",
                            units="degrees_north",
                            standard_name="latitude",
                        ),
                    ),
                    "lon": (
                        ["lon"],
                        longitude,
                        dict(
                            axis="X",
                            units="degrees_east",
                            standard_name="longitude",
                        ),
                    ),
                }
            )
    elif latitude.ndim == 2:
        ds_out = xr.Dataset(
            {
                "lat": (
                    ["Y", "X"],
                    latitude,
                    dict(units="degrees_north", standard_name="latitude"),
                ),
                "lon": (
                    ["Y", "X"],
                    longitude,
                    dict(units="degrees_east", standard_name="longitude"),
                ),
            }
        )
    else:
        raise IndexError(f"{latitude.ndim}D latitude/longitude arrays not supported.")

    return ds_out


def select(
    da,
    longitude=None,
    latitude=None,
    T=None,
    Z=None,
    iT=None,
    iZ=None,
    extrap=False,
    extrap_val=None,
    horizontal_interp: bool = True,
    horizontal_interp_code: str = "xesmf",
    use_projection: bool = True,
    triangulation = None,
    mask: Optional[DataArray] = None,
    use_xoak: bool = False,
    vertical_interp: bool = False,
    xgcm_grid = None,
    locstream=False,
    weights=None,
    make_time_series=False,
    return_info: bool = False,
):
    """Extract output from da at location(s).

    Parameters
    ----------
    da: DataArray
        DataArray from which to extract data.
    longitude, latitude: int, float, list, array (1D or 2D), DataArray, optional
        longitude(s), latitude(s) at which to return model output.
        Package `xESMF` will be used to interpolate with "bilinear" to
        these horizontal locations if horizontal_interp is True. If 
        horizontal_interp is False, then nearest neighbors will be found with 
        sel2d.
    T: datetime-like string, list of datetime-like strings, optional
        Datetime or datetimes at which to return model output.
        `xarray`'s built-in 1D interpolation will be used to calculate.
        To selection time in any way, use either this keyword argument
        or `iT`, but not both simultaneously.
    Z: int, float, list, optional
        Depth(s) at which to return model output.
        `xarray`'s built-in 1D interpolation will be used to calculate.
        To selection depth in any way, use either this keyword argument
        or `iZ`, but not both simultaneously.
    iT: int or list of ints, optional
        Index of time in time coordinate to select using `.isel`.
        To selection time in any way, use either this keyword argument
        or `T`, but not both simultaneously.
    iZ: int or list of ints, optional
        Index of depth in depth coordinate to select using `.isel`.
        To selection depth in any way, use either this keyword argument
        or `Z`, but not both simultaneously.
    extrap: bool, optional
        Whether or not to extrapolate outside the available domain.
        If False, will return 0 by default outside the domain, or
        optionally `extrap_value`. If True, will use
        `extrap_method = "nearest_s2d"` to extrapolate.
    extrap_val: int, float, optional
        If `extrap==False`, values outside domain will be returned as 0,
        or as `extrap_value` if input.
    horizontal_interp: bool
        True to interpolate, False to find nearest neighbor with sel2d.
    horizontal_interp_code: str
        Default "xesmf" to use package ``xESMF`` for horizontal interpolation, which is probably better if you need to interpolate to many points. To use ``xESMF`` you have install it as an optional dependency. Input "tree" to use BallTree to find nearest 3 neighbors and interpolate using barycentric coordinates. This has been tested for interpolating to 3 locations so far. Input "delaunay" to use a delaunay triangulation to find the nearest triangle points and interpolate the same as with "tree" using barycentric coordinates. This should be faster when you have more points to interpolate to, especially if you save and reuse the triangulation.
    use_projection: bool
        If True, project coordinates using Albers Equal Area projection for running interpolation. Only applies for ``horizontal_interp_code="tree"`` and ``horizontal_interp_code="delaunay"``. Otherwise keep in long/lat.
    triangulation:
        Input to reuse Delaunary triangulation made up of the lon/lat grid for interpolation. Use with ``horizontal_interp_code="delaunay"``.
    mask : DataArray, optional
        Mask associated with da to be used with nearest neighbor search, if desired, so that
        masked elements are not returned.
    use_xoak : bool
        If True, use xoak to find nearest 1 point. If False, use BallTree directly to find distances and nearest 4 points.
    vertical_interp : bool
        If True, interpolate in depth to input Z values. If False, use `method="nearest"` to return nearest depths to Z.
    xgcm_grid
        If you want to interpolate vertically in more than 1 dimension, you need to have an xGCM grid defined that tells xgcm how to interpret the grid vertically. You can get this for known models using, e.g., ``preprocess_roms_grid()``.
    locstream: boolean, optional
        Which type of interpolation to do:

        * False: 2D array of points with 1 dimension the lons and the other dimension the lats.
        * True: lons/lats as unstructured coordinate pairs (in xESMF language, LocStream).
    weights: xESMF netCDF file path, DataArray, optional
        If a weights file or array exists you can pass it as an argument here and it will
        be re-used.
    make_time_series: bool
        If True, use advanced indexing to select final result down to time series matching da.cf["T"] length.
    return_info: bool
        If True, return a dict of extra information that depends on what processes were run.

    Returns
    -------
    DataArray of interpolated and/or selected values from da, and kwargs_out which contains different things depending on the code that was run.

    Examples
    --------

    Select a single grid point.

    >>> longitude = 100
    >>> latitude = 10
    >>> iZ = 0
    >>> iT = 0
    >>> varname = 'u'
    >>> kwargs = dict(da=da, longitude=longitude, latitude=latitude, iT=iT, iZ=iZ, varname=varname)
    >>> da_out = em.select(**kwargs)
    """
    
    kwargs_out = {}

    # Must select or interpolate for depth and time.
    # - i.e. One cannot run in both Z and iZ mode, same for T/iT
    if (Z is not None) and (iZ is not None):
        raise ValueError("Cannot specify both Z and iZ.")
    if (T is not None) and (iT is not None):
        raise ValueError("Cannot specify both T and iT.")

    if (longitude is not None) and (latitude is not None):
        # Must convert scalars to lists because 0D lat/lon arrays are not supported.
        if isinstance(longitude, Number):
            longitude = [longitude]
        if isinstance(latitude, Number):
            latitude = [latitude]
        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)
        # # If longitude and latitude are input, perform horizontal interpolation
        # horizontal_interp = True
    # else:
    #     horizontal_interp = False

    # If extrapolating, define method
    if extrap:
        extrap_method = "nearest_s2d"
    else:
        extrap_method = None

    # Horizontal interpolation #
    # Verify interpolated points in domain if not extrapolating.
    if horizontal_interp and not extrap:
        if (
            longitude.min() < da.cf["longitude"].min()
            or longitude.max() > da.cf["longitude"].max()
        ):
            raise ValueError(
                "Longitude outside of available domain. "
                "Use extrap=True to extrapolate."
            )
        if (
            latitude.min() < da.cf["latitude"].min()
            or latitude.max() > da.cf["latitude"].max()
        ):
            raise ValueError(
                "Latitude outside of available domain. "
                "Use extrap=True to extrapolate."
            )

    # Perform interpolation
    if horizontal_interp:

        from time import time
        start_time = time()

        if XESMF_AVAILABLE and horizontal_interp_code == "xesmf":
            # create Dataset to interpolate to
            ds_out = make_output_ds(longitude, latitude, locstream=locstream)

            da, weights = interp_multi_dim(
                da,
                ds_out,
                # T=T,
                # Z=Z,
                # iT=iT,
                # iZ=iZ,
                extrap_method=extrap_method,
                locstream=locstream,
                weights=weights,
            )
            kwargs_out["weights"] = weights

        elif horizontal_interp_code == "delaunay":

            # This might be faster for interpolation between a few points and many points
            # get points for trianlges from delaunay
            from scipy.spatial import Delaunay

            # calculate triangulation
            X = np.stack([np.ravel(c) for c in [da.cf["longitude"], da.cf["latitude"]]]).T
            if triangulation is not None:
                tri = triangulation
            else:
                tri = Delaunay(X) 

            # save triangulation for potential future use
            kwargs_out["tri"] = tri

            # prep points to interpolate to
            p = np.stack([np.ravel(c) for c in [longitude, latitude]]).T
            
            # itri are the triangle indices containing the points we are interpolating to
            itri = tri.find_simplex(p)
            
            # xs, ys are the vertices of the triangles around the points, npts x 3
            # xs, ys, x, y here are actually longs and lats but will be overwritten below if 
            # use_projection
            xs, ys = X[tri.simplices[itri]][:,:,0], X[tri.simplices[itri]][:,:,1]
            x, y = longitude, latitude
            iys, ixs = np.unravel_index(tri.simplices[itri], da.cf["longitude"].shape)

            if use_projection:
                # convert to projected coordinates
                import pyproj
                min_lat, max_lat = float(da.cf["latitude"].min()), float(da.cf["latitude"].max())
                mean_lon = float(da.cf["longitude"].mean())
                proj = pyproj.Proj(f'+proj=aea +lat_1={max_lat} +lat_2={min_lat} +lon_0={mean_lon}')

                xs, ys = proj(xs, ys)
                x, y = proj(longitude, latitude)

            lam = calc_barycentric(x, y, xs, ys)
            # interp_coords are the coords and indices that went into the interpolation
            da, interp_coords = interp_with_barycentric(da, ixs, iys, lam)
            kwargs_out["interp_coords"] = interp_coords
             
        elif horizontal_interp_code == "tree":
            
            # get points from balltree
            npts = len(longitude)
            
            # distances, iys_init, ixs_init are npts x k=6
            distances, (iys_init, ixs_init) = tree_query(da.cf["longitude"], da.cf["latitude"], longitude, latitude, k=4)
            
            if use_projection:
                import pyproj

                # convert to projected coordinates
                min_lat, max_lat = float(da.cf["latitude"].min()), float(da.cf["latitude"].max())
                mean_lon = float(da.cf["longitude"].mean())
                proj = pyproj.Proj(f'+proj=aea +lat_1={max_lat} +lat_2={min_lat} +lon_0={mean_lon}')
            
                # # triangle points, projected, npts x 3
                # xs_init, ys_init = proj(da.cf["longitude"].values[iys_init, ixs_init], da.cf["latitude"].values[iys_init, ixs_init])
                # points to find
                x, y = proj(longitude, latitude)
            else:
                x, y = longitude, latitude
            
            # Choose the first three points that actually make a triangle around each point
            from shapely.geometry import Polygon, Point
            
            # xs, ys store the 3 nearest points making up triangles around each of npts
            # npts x 3
            xs, ys = np.empty((npts, 3)), np.empty((npts, 3))
            ixs, iys = np.empty((npts, 3), dtype=int), np.empty((npts, 3), dtype=int)
            
            # check combinations of 3 points for each npt to find triangle
            import itertools
            for ipt in range(npts):
                # unique combinations of 3 of the nearest points to ipt, by index
                ix_combinations = list(itertools.combinations(ixs_init[ipt], 3))
                iy_combinations = list(itertools.combinations(iys_init[ipt], 3))
        
                # point we are interpolating to
                # if use_projection, in projected coordinates, otherwise in lon/lat
                pt = Point(x[ipt], y[ipt])

                def pt_in_itriangle(ix, iy):
                    # projected triangle points for combinations
                    xs, ys = da.cf["longitude"].values[iy, ix], da.cf["latitude"].values[iy, ix]
                    return Polygon(np.stack([xs, ys]).T).contains(pt)
                
                def pt_in_itriangle_proj(ix, iy):
                    # projected triangle points for combinations
                    xs, ys = proj(da.cf["longitude"].values[iy, ix], da.cf["latitude"].values[iy, ix])
                    return Polygon(np.stack([xs, ys]).T).contains(pt)
                
                i=0
                if use_projection:
                    while not pt_in_itriangle_proj(ix_combinations[i], iy_combinations[i]):
                        i += 1
                else:
                    while not pt_in_itriangle(ix_combinations[i], iy_combinations[i]):
                        i += 1

                # also update ixs, iys
                ixs[ipt], iys[ipt] = ix_combinations[i], iy_combinations[i]
                if use_projection:
                    xs[ipt], ys[ipt] = proj(da.cf["longitude"].values[iys[ipt], ixs[ipt]], da.cf["latitude"].values[iys[ipt], ixs[ipt]])
                else:
                    xs[ipt], ys[ipt] = da.cf["longitude"].values[iys[ipt], ixs[ipt]], da.cf["latitude"].values[iys[ipt], ixs[ipt]]
            
            lam = calc_barycentric(x, y, xs, ys)
            # interp_coords are the coords and indices that went into the interpolation
            da, interp_coords = interp_with_barycentric(da, ixs, iys, lam)
            kwargs_out["interp_coords"] = interp_coords

        else:
            raise ModuleNotFoundError(
                "xESMF is not available so horizontal interpolation in 2D cannot be performed."
            )
        
        end_time = time()
        print("time: ", start_time-end_time)
    # nearest neighbor instead
    elif not horizontal_interp and longitude is not None and latitude is not None:
        da, k_out = da.em.sel2dcf(longitude=longitude, latitude=latitude, mask=mask, use_xoak=use_xoak, ) 
        kwargs_out["distances"] = k_out["distances"]
        kwargs_out.update(k_out)
    else:
        print("no horizontal grid changes")

    if iT is not None:
        with xr.set_options(keep_attrs=True):
            da = da.cf.isel(T=iT)

    elif T is not None:
        with xr.set_options(keep_attrs=True):
            da = da.cf.interp(T=T)

    # Time and depth interpolation or iselection #
    if iZ is not None:
        with xr.set_options(keep_attrs=True):
            da = da.cf.isel(Z=iZ)

    elif Z is not None:
        
        # deal with interpolation in Z separately
        if vertical_interp:
            # can do interpolation in depth for any number of dimensions if the
            # vertical coord is 1d
            if da.cf["vertical"].ndim == 1:
                if da.cf["vertical"].name in ("s_rho", "s_w"):
                    raise UserWarning(f"The dimension identified as the vertical coordinate is {da.cf['vertical'].name} which might not be correct.")
                
                da = da.cf.interp(vertical=Z)
            
            else:
                # # use xgcm
                # from xgcm import Grid
                # grid = Grid(ds, coords={'Z': {'center':'s_rho', 'outer':'s_w'},
                #                         },
                #                 periodic=False
                #             )
                
                # need "grid" from xgcm set up in preprocessing 
                if xgcm_grid is None:
                    raise KeyError("Need xgcm 'grid' object set up and input as ``xgcm_grid``.")
                z_attrs = da.cf["vertical"].attrs
                z_attrs.update({"axis": "Z"})
                zkey = da.cf["vertical"].name
                da = xgcm_grid.transform(da, 'Z', np.array(Z), target_data=da.cf["vertical"], method='linear')
                da[zkey].attrs = z_attrs
                da = order(da)  # reorder dimensions to convention
            

            # # if the vertical coord is greater than 1D, can only do restricted interpolation
            # # at the moment
            # else:
            #     da = da.squeeze()
            #     if len(da.dims) == 1 and da.cf["Z"].name in da.dims:
            #         da = da.swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
            #         da = da.cf.interp(vertical=Z)
            #     elif len(da.dims) == 2 and da.cf["Z"].name in da.dims:
            #         # loop over other dimension
            #         dim_var_name = list(set(da.dims) - set([da.cf["Z"].name]))[0]
            #         new_da = []
            #         for i in range(len(da[dim_var_name])):
            #             new_da.append(
            #                 da.isel({dim_var_name: i})
            #                 .swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
            #                 .cf.interp(vertical=Z)
            #             )
            #         da = xr.concat(new_da, dim=dim_var_name)
            #     elif len(da.dims) > 2:
            #         # need to implement (x)isoslice here
            #         raise NotImplementedError(
            #             "Currently it is not possible to interpolate in depth with more than 1 other (time) dimension."
            #         )
        # or just grab nearest
        else:
            # this works if Z is depth instead of sigma
            da = da.cf.sel(Z=Z, method="nearest")
            # this is for needing to use vertical to access depths
            # da = da.squeeze()
            # if da.cf["vertical"].ndim == 1 and da.cf["vertical"].name in da.coords:
            #     da = da.swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
            #     da = da.cf.sel(Z=Z)
            # elif len(da.dims) == 2 and da.cf["Z"].name in da.dims:
            #     # loop over other dimension
            #     dim_var_name = list(set(da.dims) - set([da.cf["Z"].name]))[0]
            #     new_da = []
            #     for i in range(len(da[dim_var_name])):
            #         new_da.append(
            #             da.isel({dim_var_name: i})
            #             .swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
            #             .cf.interp(vertical=Z)
            #         )
            #     da = xr.concat(new_da, dim=dim_var_name)
            
    # advanced indexing to select all assuming coherent time series
    # make sure len of each dimension matches
    if make_time_series:
        
        dims_to_index = [da.cf["T"].name]
        ntimes = len(da.cf["T"])
        for axis, var_names in da.cf.axes.items():
            for var_name in var_names:
                if len(da[var_name]) == ntimes and var_name in da.dims:
                    dims_to_index.append(var_name)
                    
        # use time dim as dim for all since treating as time series
        indexer = {dim: xr.DataArray(np.arange(0,ntimes), dims=da.cf["T"].name) for dim in dims_to_index}
        da = da.isel(indexer)

    if extrap_val is not None:
        # returns 0 outside the domain by default. Assumes that no other values are exactly 0
        # and replaces all 0's with extrap_val if chosen.
        da = da.where(da != 0, extrap_val)

    if return_info:
        return da.squeeze(), kwargs_out
    else:
        return da.squeeze()


def sel2d(
    var,
    mask: Optional[DataArray] = None,
    use_xoak: bool = True,
    return_info: bool = False,
    **kwargs,
):
    """Find the value of the var at closest location to inputs, optionally respecting mask.

    This is meant to mimic `xarray` `.sel()` in API and idea, except that the horizontal selection is done for 2D coordinates instead of 1D coordinates, since `xarray` cannot yet handle 2D coordinates. This wraps `xoak`.

    Order of inputs is important:
    - use: lonname, latname, other inputs
    - or: latname, lonname, other inputs
    will also work if lonname, latname include "lon" and "lat" in their names, respectively.

    Like in `xarray.sel()`, input values can be numbers, lists, or arrays, and arrays can be single or multidimensional. For longitude and latitude, however, input values cannot be slices.

    Can also pass through `xarray.sel()` information for other dimension selections.

    Optionally input mask so that if requested lon/lat is on land, the nearest valid model point will be returned. Otherwise nan's will be returned. If requested lon/lat is outside domain but not on land, the nearest model output will be returned regardless.

    Parameters
    ----------
    var: DataArray, Dataset
        Container from which to extract data. For a DataArray, it is better to input a separate object instead of a Dataset with variable specified so that it remembers the index that is calculated. That is, use:
        >>> da = ds.variable
        >>> em.sel2d(da, ...)
        instead of `ds.variable` directly. Then subsequent calls will be faster. See `xoak` for more information.
        A Dataset will "remember" the index calculated for whichever grid coordinates were first requested and subsequently run faster for requests on that grid (and not run for other grids).
    mask : DataArray, optional
        If input, mask is applied to lon/lat so that if requested lon/lat is on land, the nearest valid model point will be returned. Otherwise nan's will be returned. If requested lon/lat is outside domain but not on land, the nearest model output will be returned regardless.
    use_xoak : bool
        If True, use xoak to find nearest 1 point. If False, use BallTree directly to find distances and nearest 4 points.
    return_info: bool
        If True, return a dict of extra information that depends on what processes were run.

    Returns
    -------
    An xarray object of the same type as input as var which is selected in horizontal coordinates to input locations and, in input, to time and vertical selections. If not selected, other dimensions are brought along. Other items returned in kwargs include:
    
    * distances: the distances from the requested points to the returned nearest points

    Notes
    -----
    If var is a Dataset and contains more than one horizontal grid, the lonname, latname you use should match the variables you want to be able to access at the desired locations.

    Examples
    --------
    Select grid node of DataArray nearest to location (-96, 27). The DataArray `ds.temp` has coordinates `lon_rho` and `lat_rho`:

    >>> da = ds.temp
    >>> em.sel2d(da, lon_rho=-96, lat_rho=27)

    To additionally select from the time and depth dimensions, call as:

    >>> em.sel2d(da, lon_rho=-96, lat_rho=27, time='2022-07-19', s_rho=0.0)

    You may also input other keywords to pass onto `xarray.sel()`:

    >>> em.sel2d(da, lon_rho=-96, lat_rho=27, s_rho=0.0, method='nearest')
    """

    # if dask-backed, read lon/lat into memory for faster working later
    if var.cf["longitude"].chunks is not None:
        var[var.cf["longitude"].name] = var.cf["longitude"].load()
    if var.cf["latitude"].chunks is not None:
        var[var.cf["latitude"].name] = var.cf["latitude"].load()

    # assign input lon/lat coord names
    kwargs_iter = iter(kwargs)
    # first key is assumed longitude unless has "lat" in the name
    # second key is assumed latitude unless has "lon" in the name
    key1 = next(kwargs_iter)
    key2 = next(kwargs_iter)
    lonname, latname = key1, key2
    if ("lat" in key1) and ("lon" in key2):
        latname = key1
        lonname = key2

    lons, lats = kwargs[lonname], kwargs[latname]

    # remove lon/lat info from kwargs for passing it later
    kwargs.pop(key1)
    kwargs.pop(key2)

    # Make a Dataset out of input lons/lats
    # make sure everything is a numpy array
    if isinstance(lons, Number) and isinstance(lats, Number):
        lons, lats = np.array([lons]), np.array([lats])
    elif isinstance(lons, list) and isinstance(lats, list):
        lons, lats = np.array(lons), np.array(lats)
        
    if use_xoak:

        # 1D or 2D
        if lons.ndim == lats.ndim == 1:
            dims = ("loc",)
        elif lons.ndim == lats.ndim == 2:
            dims = ("loc_y", "loc_x")
        # else: Raise exception

        # create Dataset
        ds_to_find = xr.Dataset(
            {
                "lat_to_find": (dims, lats, {"standard_name": "latitude"}),
                "lon_to_find": (dims, lons, {"standard_name": "longitude"}),
            }
        )

        if mask is not None:
            
            # if dask-backed, read into memory
            if mask.chunks is not None:
                mask = mask.load()

            # Assume mask is 2D — but not true for wetting/drying
            # import pdb; pdb.set_trace()
            # find indices representing mask
            eta, xi = np.where(mask.values)

            # make advanced indexer to flatten arrays
            var_flat = var.cf.isel(
                X=xr.DataArray(xi, dims="loc"), Y=xr.DataArray(eta, dims="loc")
            )

            var = var_flat.copy()

        if var.xoak.index is None:
            var.xoak.set_index([latname, lonname], "sklearn_geo_balltree")
        elif (latname, lonname) != var.xoak._index_coords:
            raise ValueError(
                f"Index has been built for grid with coords {var.xoak._index_coords} but coord names input are ({latname}, {lonname})."
            )
        elif var.xoak.index is not None:
            pass
        else:
            warnings.warn(
                "Maybe a mask is not present or being properly identified in var. You could use `use_mask=False`.",
                RuntimeWarning,
            )

        # perform selection
        output = var.xoak.sel(
            {latname: ds_to_find.lat_to_find, lonname: ds_to_find.lon_to_find}
        )

        # # this version is for the updates in xoak
        # output = var.xoak.sel(
        #     {latname: ds_to_find.lat_to_find, lonname: ds_to_find.lon_to_find},
        #     distances_name=distances_name,
        # )
        # output[distances_name] *= 6371  # convert from radians to km

        # only calculate distances this way (outside of xoak itself) if not doing 2D since we just need distances
        # right now for OMSA and don't want a separate soln from xoak for this problem
        if ds_to_find.lat_to_find.ndim > 1:
            with xr.set_options(keep_attrs=True):
                if return_info:
                    # no info to return but set up for future use
                    kwargs_out = {}
                    return output.sel(**kwargs), kwargs_out
                else:
                    return output.sel(**kwargs)

        # distances between input points and nearest points - this won'tbe needed with new version of xoak once merged
        # * 6371 to convert from radians to km
        index = var.xoak._index
        if isinstance(index, tuple):
            index = index[0]
        query = index.query(np.array([*zip(lats, lons)]))
        from dask.delayed import Delayed
        if isinstance(query, Delayed):
            query = query.compute()
        distances = query["distances"][:, 0] * 6371
        iflat = query["indices"][:,0].tolist()[0]
        if mask is None and var.cf["X"].ndim == 1:
            xi, eta = np.meshgrid(var.cf["X"], var.cf["Y"])
            xi, eta = xi.flatten(), eta.flatten()

        # this is probably wrong if you aren't looking for matches at a single point
        ixi = xi[iflat]
        ieta = eta[iflat]
        
        if isinstance(distances, Delayed):
            distances = distances.compute()
        if not isinstance(output, Dataset):
            output = output.to_dataset()
        attrs = {"units": "km"}
        indexer_dim = ds_to_find.lat_to_find.dims
        indexer_shape = ds_to_find.lat_to_find.shape
        distances = xr.Variable(
            indexer_dim, distances.reshape(indexer_shape), attrs
        )
        # kwargs["distances"] = distances

        with xr.set_options(keep_attrs=True):
            output = output.sel(**kwargs)
            kwargs["distances"] = distances
            if return_info:
                return output, kwargs
            # return output.sel(**kwargs), kwargs
            else:
                return output
    
    else:
        
        # currently lons, lats 1D only
        
        # if no mask, assume user just wants 1 nearest point to each input lons/lats pair
        # probably should expand this later to be more generic
        if mask is None:
            k = 1
        # if user inputs mask, use it to only return the nearest point that is active
        # so, find nearest 30 points to have options
        else:
            k = 30
        
        distances, (iys, ixs) = tree_query(var[lonname], var[latname], lons, lats, k=k)
        
        # sort mask such that active elements are preferentially to the left in a 2D array
        if mask.values[iys,ixs].sum() == 0:
            raise ValueError("all found values are masked!")

        isorted_mask = np.argsort(-mask.values[iys,ixs], axis=-1)
        # sort the ixs and iys according to this sorting so that if there are unmasked indices, 
        # they are leftmost also, and we will use the leftmost values.
        ixs_brought_along = np.take_along_axis(ixs, isorted_mask, axis=1)
        iys_brought_along = np.take_along_axis(iys, isorted_mask, axis=1)
        distances_brought_along = np.take_along_axis(distances, isorted_mask, axis=1)
        ixs0 = ixs_brought_along[:,0]
        iys0 = iys_brought_along[:,0]
        distances0 = distances_brought_along[:,0]

        # ipoint = 0
        # # if inds[0] is masked, check the next until finding one that isn't mask
        # # if no mask provided, just take first point
        # if mask is not None:
        #     def is_masked(iy, ix):
        #         return int(mask[iy, ix]) == 0
            
        #     while is_masked(iys[ipoint], ixs[ipoint]):
        #         ipoint += 1

        kwargs[var.cf["Y"].name], kwargs[var.cf["X"].name] = iys0, ixs0
        
        dims = ("npts",)  
        var_out = var.cf.isel(X=xr.DataArray(ixs0, dims=dims), Y=xr.DataArray(iys0, dims=dims))
        # add "X" axis to npts
        var_out["npts"] = ("npts", var_out.npts.values, {"axis": "X"})
        
        kwargs["distances"] = distances0* 6371

        with xr.set_options(keep_attrs=True):
            if return_info:
                return var_out, kwargs
            else:
                return var_out


def sel2dcf(
    var,
    mask: Optional[DataArray] = None,
    return_info: bool = False,
    **kwargs,
):
    """Find nearest value(s) on 2D horizontal grid using cf-xarray names.

    Use "longitude" and "latitude" for those coordinate names.

    You can input a combination of variable names and cf-xarray names for time and z dimensions. Order of the input names doesn't matter (unlike in `em.sel2d()`).

    See `sel2d` for full docs. This wraps that function but will use cf-xarray standard names.

    Examples
    --------
    Select grid node of DataArray nearest to location (-96, 27):

    >>> da = ds.temp
    >>> em.sel2d(da, longitude=-96, latitude=27)

    To additionally select from the time and depth dimensions, call as:

    >>> em.sel2d(da, longitude=-96, latitude=27, T='2022-07-19', Z=0.0)

    You may also input other keywords to pass onto `xarray.sel()`:

    >>> em.sel2d(da, longitude=-96, latitude=27, Z=0.0, method='nearest')
    """

    lons, lats = kwargs["longitude"], kwargs["latitude"]

    # use cf-xarray to get lon/lat key names
    try:
        latname = var.cf["latitude"].name
        lonname = var.cf["longitude"].name
    except KeyError:
        print(
            "cf-xarray cannot determine variable name for longitude and latitude. Instead, use `sel2d()` and input the coordinate names specifically."
        )

    kwargs.pop("longitude")
    kwargs.pop("latitude")

    # need to maintain order
    new_kwargs = {lonname: lons, latname: lats}

    if "T" in kwargs:
        tname = var.cf["T"].name
        new_kwargs[tname] = kwargs["T"]
        kwargs.pop("T")
    if "Z" in kwargs:
        zname = var.cf["Z"].name
        new_kwargs[zname] = kwargs["Z"]
        kwargs.pop("Z")

    new_kwargs.update(kwargs)

    return sel2d(var, mask=mask, return_info=return_info, **new_kwargs)


def selZ(var, depths):
    """Select nearest point in depth.

    This is for when the depth coordinates are 4D, for example in ROMS. This is not a smart algorithm but should be replaced in the future. For now, it simply loops over the times for a single loc and uses xarray's `.sel` with `method='nearest'` to be able to select the closest model output to the desired depth. The resulting DataArray shape is odd if there is more than one depth because depth changes in time.

    Parameters
    ----------
    var: DataArray
        Container from which to extract data.
    depths: int, float, list, optional
        Depth(s) at which to return model output.
        `xarray`'s built-in 1D nearest selection will be used to calculate.
    """

    # # Make a Dataset out of input lons/lats
    # # make sure everything is a numpy array
    # if isinstance(depths, Number):
    #     depths = np.array([depths])
    # elif isinstance(depths, list):
    #     depths = np.array(depths)

    if var.ndim == 1:
        var = var.swap_dims({var.cf["Z"].name: var.cf["vertical"].name})
        out = var.cf.sel(vertical=depths, method="nearest")
    elif var.ndim == 2:
        # loop over other dimension
        dim_var_name = list(set(var.dims) - set([var.cf["Z"].name]))[0]
        new_results = []
        for i in range(len(var[dim_var_name])):
            new_results.append(
                var.isel({dim_var_name: i})
                .swap_dims({var.cf["Z"].name: var.cf["vertical"].name})
                .cf.sel(vertical=depths, method="nearest")
            )
        out = xr.concat(new_results, dim=dim_var_name)
    else:
        raise NotImplementedError("Sorry only works for 1D and 2D so far.")

    return out
