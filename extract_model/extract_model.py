"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""

import cartopy.geodesic
import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
import xesmf as xe


# try:
# except ImportError:
#     warnings.warn(
#         "cartopy is not installed, so `sel2d` and `argsel2d` will not run.",
#         ImportWarning,
#     )


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
    regridder=None,
):
    """Extract output from da at location(s).

    Parameters
    ----------
    da: DataArray
        Property to select model output from.
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
        or as `extra_value` if input.
    locstream: boolean, optional
        Which type of interpolation to do:

        * False: 2D array of points with 1 dimension the lons and the other dimension the lats.
        * True: lons/lats as unstructured coordinate pairs (in xESMF language, LocStream).
    regridder: xESMF regridder object
        If this interpolation setup has been performed before and regridder saved,
        you can input it to save time. This is the same as the weights.


    Returns
    -------
    DataArray of interpolated and/or selected values from da.

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

    # can't run in both Z and iZ mode, same for T/iT
    assert not ((Z is not None) and (iZ is not None))
    assert not ((T is not None) and (iT is not None))

    if (longitude is not None) and (latitude is not None):
        if (isinstance(longitude, int)) or (isinstance(longitude, float)):
            longitude = [longitude]
        if (isinstance(latitude, int)) or (isinstance(latitude, float)):
            latitude = [latitude]
        latitude = np.asarray(latitude)
        longitude = np.asarray(longitude)

    if extrap:
        extrap_method = "nearest_s2d"
    else:
        extrap_method = None

    if (not extrap) and ((longitude is not None) and (latitude is not None)):
        assertion = "the input longitude range is outside the model domain"
        assert (longitude.min() >= da.cf["longitude"].min()) and (
            longitude.max() <= da.cf["longitude"].max()
        ), assertion
        assertion = "the input latitude range is outside the model domain"
        assert (latitude.min() >= da.cf["latitude"].min()) and (
            latitude.max() <= da.cf["latitude"].max()
        ), assertion

    # Horizontal interpolation #
    # grid of lon/lat to interpolate to, with desired ending attributes
    if (longitude is not None) and (latitude is not None):

        if latitude.ndim == 1:
            if locstream:
                # for locstream need a single dimension (in this case called "loc")
                da_out = xr.Dataset(
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
                da_out = xr.Dataset(
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
            da_out = xr.Dataset(
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

        if regridder is None:
            # set up regridder, which would work for multiple interpolations if desired
            regridder = xe.Regridder(
                da,
                da_out,
                "bilinear",
                extrap_method=extrap_method,
                locstream_out=locstream,
                ignore_degenerate=True,
            )

        # do regridding
        da_int = regridder(da, keep_attrs=True)
    else:
        da_int = da

    # get z coordinates to go with interpolated output if not available
    if "vertical" in da.cf.coords:
        zkey = da.cf["vertical"].name

        # only need to interpolate z coordinates if they are not 1D
        if da[zkey].ndim > 1:
            zint = regridder(da[zkey], keep_attrs=True)

            # add coords
            da_int = da_int.assign_coords({zkey: zint})

    if iT is not None:
        with xr.set_options(keep_attrs=True):
            da_int = da_int.cf.isel(T=iT)

    elif T is not None:
        with xr.set_options(keep_attrs=True):
            da_int = da_int.cf.interp(T=T)

    # Time and depth interpolation or iselection #
    if iZ is not None:
        with xr.set_options(keep_attrs=True):
            da_int = da_int.cf.isel(Z=iZ)

    # deal with interpolation in Z separately
    elif Z is not None:

        # can do interpolation in depth for any number of dimensions if the
        # vertical coord is 1d
        if da_int.cf["vertical"].ndim == 1:
            da_int = da_int.cf.interp(vertical=Z)

        # if the vertical coord is greater than 1D, can only do restricted interpolation
        # at the moment
        else:
            da_int = da_int.squeeze()
            if len(da_int.dims) == 1 and da_int.cf["Z"].name in da_int.dims:
                da_int = da_int.swap_dims(
                    {da_int.cf["Z"].name: da_int.cf["vertical"].name}
                )
                da_int = da_int.cf.interp(vertical=Z)
            elif len(da_int.dims) == 2 and da_int.cf["Z"].name in da_int.dims:
                # loop over other dimension
                dim_var_name = list(set(da_int.dims) - set([da_int.cf["Z"].name]))[0]
                new_da = []
                for i in range(len(da_int[dim_var_name])):
                    new_da.append(
                        da_int.isel({dim_var_name: i})
                        .swap_dims({da_int.cf["Z"].name: da_int.cf["vertical"].name})
                        .cf.interp(vertical=Z)
                    )
                da_int = xr.concat(new_da, dim=dim_var_name)
            elif len(da_int.dims) > 2:
                # need to implement (x)isoslice here
                raise NotImplementedError(
                    "Currently it is not possible to interpolate in depth with more than 1 other (time) dimension."
                )

    if extrap_val is not None:
        # returns 0 outside the domain by default. Assumes that no other values are exactly 0
        # and replaces all 0's with extrap_val if chosen.
        da_int = da_int.where(da_int != 0, extrap_val)

    return da_int.squeeze(), regridder


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
