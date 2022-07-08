"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""

import warnings

from numbers import Number

import cartopy.geodesic
import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr


try:
    import xesmf as xe

    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    warnings.warn("xESMF not found. Interpolation will not work.")

# try:
# except ImportError:
#     warnings.warn(
#         "cartopy is not installed, so `sel2d` and `argsel2d` will not run.",
#         ImportWarning,
#     )


def interp_multi_dim(
    da,
    da_out=None,
    T=None,
    Z=None,
    iT=None,
    iZ=None,
    extrap_method=None,
    locstream=False,
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
        raise ModuleNotFoundError(
            "xESMF is not available so horizontal interpolation in 2D cannot be performed."
        )

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
    locstream=False,
    weights=None,
):
    """Extract output from da at location(s).

    Parameters
    ----------
    da: DataArray
        DataArray from which to extract data.
    longitude, latitude: int, float, list, array (1D or 2D), DataArray, optional
        longitude(s), latitude(s) at which to return model output.
        Package `xESMF` will be used to interpolate with "bilinear" to
        these horizontal locations.
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
    locstream: boolean, optional
        Which type of interpolation to do:

        * False: 2D array of points with 1 dimension the lons and the other dimension the lats.
        * True: lons/lats as unstructured coordinate pairs (in xESMF language, LocStream).
    weights: xESMF netCDF file path, DataArray, optional
        If a weights file or array exists you can pass it as an argument here and it will
        be re-used.

    Returns
    -------
    DataArray of interpolated and/or selected values from da, and array of weights.

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
        # If longitude and latitude are input, perform horizontal interpolation
        horizontal_interp = True
    else:
        horizontal_interp = False

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
        # create Dataset to interpolate to
        ds_out = make_output_ds(longitude, latitude, locstream=locstream)

        if XESMF_AVAILABLE:
            da, weights = interp_multi_dim(
                da,
                ds_out,
                T=T,
                Z=Z,
                iT=iT,
                iZ=iZ,
                extrap_method=extrap_method,
                locstream=locstream,
                weights=weights,
            )
        else:
            raise ModuleNotFoundError(
                "xESMF is not available so horizontal interpolation in 2D cannot be performed."
            )

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

    # deal with interpolation in Z separately
    elif Z is not None:

        # can do interpolation in depth for any number of dimensions if the
        # vertical coord is 1d
        if da.cf["vertical"].ndim == 1:
            da = da.cf.interp(vertical=Z)

        # if the vertical coord is greater than 1D, can only do restricted interpolation
        # at the moment
        else:
            da = da.squeeze()
            if len(da.dims) == 1 and da.cf["Z"].name in da.dims:
                da = da.swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
                da = da.cf.interp(vertical=Z)
            elif len(da.dims) == 2 and da.cf["Z"].name in da.dims:
                # loop over other dimension
                dim_var_name = list(set(da.dims) - set([da.cf["Z"].name]))[0]
                new_da = []
                for i in range(len(da[dim_var_name])):
                    new_da.append(
                        da.isel({dim_var_name: i})
                        .swap_dims({da.cf["Z"].name: da.cf["vertical"].name})
                        .cf.interp(vertical=Z)
                    )
                da = xr.concat(new_da, dim=dim_var_name)
            elif len(da.dims) > 2:
                # need to implement (x)isoslice here
                raise NotImplementedError(
                    "Currently it is not possible to interpolate in depth with more than 1 other (time) dimension."
                )

    if extrap_val is not None:
        # returns 0 outside the domain by default. Assumes that no other values are exactly 0
        # and replaces all 0's with extrap_val if chosen.
        da = da.where(da != 0, extrap_val)

    return da.squeeze(), weights


def argsel2d(lons, lats, lon0, lat0):
    """Find the indices of coordinate pair closest to another point.

    Inputs
    ------
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.

    Returns
    -------
    Index or indices of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. Number of dimensions of
    returned indices will correspond to the shape of input lons.

    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.

    Example usage
    -------------
    >>> em.argsel2d(ds.lon_rho, ds.lat_rho, -96, 27)
    """

    # need to address that lon/lat could 1d or 2d
    if np.asarray(lons).ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    # input lons and lats can be multidimensional and might be DataArrays or lists
    pts = list(zip(*(np.asarray(lons).flatten(), np.asarray(lats).flatten())))
    endpts = list(zip(*(np.asarray(lon0).flatten(), np.asarray(lat0).flatten())))

    G = cartopy.geodesic.Geodesic()  # set up class
    dist = np.asarray(G.inverse(pts, endpts)[:, 0])  # select distances specifically
    iclosest = abs(np.asarray(dist)).argmin()  # find indices of closest point
    # return index or indices in input array shape
    inds = np.unravel_index(iclosest, np.asarray(lons).shape)

    return inds


def sel2d(var, lons, lats, lon0, lat0, inds=None, T=None, iT=None, Z=None, iZ=None):
    """Find the value of the var at closest location to lon0,lat0.

    Can optionally include datetime-like str or index for time T axis, and/or
    vertical depth or vertical index.

    Inputs
    ------
    var: DataArray, ndarray
        Variable to operate on.
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.
    inds: list, optional
        indices of location in coordinate pairs made up of lons, lats
        that is closest to location lon0, lat0. Number of dimensions of
        returned indices will correspond to the shape of input lons.
    T: datetime-like string, list of datetime-like strings, optional
        Datetime or datetimes at which to return model output.
        `xarray`'s built-in 1D selection will be used to calculate.
        To select time in any way, use either this keyword argument
        or `iT`, but not both simultaneously.
    iT: int or list of ints, optional
        Index of time in time coordinate to select using `.isel`.
        To select time in any way, use either this keyword argument
        or `T`, but not both simultaneously.
    Z: int, float, list, optional
        Depth(s) at which to return model output.
        `xarray`'s built-in 1D interpolation will be used to calculate.
        To selection depth in any way, use either this keyword argument
        or `iZ`, but not both simultaneously.
    iZ: int or list of ints, optional
        Index of depth in depth coordinate to select using `.isel`.
        To selection depth in any way, use either this keyword argument
        or `Z`, but not both simultaneously.

    Returns
    -------
    Value in var of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. If var has other
    dimensions, they are brought along.

    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.
    This is meant to be used by the accessor to conveniently wrap
    `argsel2d`.

    Example usage
    -------------
    >>> em.sel2d(ds.temp, ds.lon_rho, ds.lat_rho, -96, 27)
    """

    # can't run in both Z and iZ mode, same for T/iT
    assert not ((Z is not None) and (iZ is not None))
    assert not ((T is not None) and (iT is not None))

    assert isinstance(var, xr.DataArray), "Input a DataArray"
    if inds is None:
        inds = argsel2d(lons, lats, lon0, lat0)

    # initialize sel and isel dicts so they are available
    isel, sel = dict(), dict()

    # selection to make
    if len(inds) == 2:  # structured models
        isel["X"] = inds[1]
        isel["Y"] = inds[0]
    elif len(inds) == 1 and "Y" in var.cf.axes:  # structured but 1D lon/lat
        isel["X"] = inds[0]
        isel["Y"] = inds[0]
    elif len(inds) == 1 and "Y" not in var.cf.axes:  # unstructured models
        isel["X"] = inds[0]
    else:
        print("I don't know this type of model.")

    # if time, include time in sel/isel dict
    if T is not None:
        sel["T"] = T
    elif iT is not None:
        isel["T"] = iT

    # if depth index, add to isel dict
    if iZ is not None:
        isel["Z"] = iZ

    # Result after T or iT, iZ, lat+lon selection
    # if depth value to be nearest to, do subsequently
    result = var.cf.sel(**sel, method="nearest").cf.isel(**isel)

    # deal with Z separately
    if Z is not None:
        if len(result.dims) == 1 and result.cf["Z"].name in result.dims:
            result = result.swap_dims({result.cf["Z"].name: result.cf["vertical"].name})
            result = result.cf.sel(vertical=Z, method="nearest")
        elif len(result.dims) == 2 and result.cf["Z"].name in result.dims:
            # loop over other dimension
            dim_var_name = list(set(result.dims) - set([result.cf["Z"].name]))[0]
            new_results = []
            for i in range(len(result[dim_var_name])):
                new_results.append(
                    result.isel({dim_var_name: i})
                    .swap_dims({result.cf["Z"].name: result.cf["vertical"].name})
                    .cf.sel(vertical=Z, method="nearest")
                )
            result = xr.concat(new_results, dim=dim_var_name)

    return result
