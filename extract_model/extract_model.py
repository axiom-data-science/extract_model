"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""

import warnings

from numbers import Number

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
import xoak  # noqa: F401


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


def sel2d(var, **kwargs):
    """Find the value of the var at closest location to inputs.

    This is meant to mimic `xarray` `.sel()` in API and idea, except that the horizontal selection is done for 2D coordinates instead of 1D coordinates, since `xarray` cannot yet handle 2D coordinates. This wraps `xoak`.

    Order of inputs is important:
    - use: lonname, latname, other inputs
    - or: latname, lonname, other inputs
    will also work if lonname, latname include "lon" and "lat" in their names, respectively.

    Like in `xarray.sel()`, input values can be numbers, lists, or arrays, and arrays can be single or multidimensional. For longitude and latitude, however, input values cannot be slices.

    Can also pass through `xarray.sel()` information for other dimension selections.

    Inputs
    ------
    var: DataArray, Dataset
        Container from which to extract data. For a DataArray, it is better to input a separate object instead of a Dataset with variable specified so that it remembers the index that is calculated. That is, use:
        >>> da = ds.variable
        >>> em.sel2d(da, ...)
        instead of `ds.variable` directly. Then subsequent calls will be faster. See `xoak` for more information.
        A Dataset will "remember" the index calculated for whichever grid coordinates were first requested and subsequently run faster for requests on that grid (and not run for other grids).

    Returns
    -------
    An xarray object of the same type as input as var which is selected in horizontal coordinates to input locations and, in input, to time and vertical selections. If not selected, other dimensions are brought along.

    Notes
    -----
    If var is a Dataset and contains more than one horizontal grid, the lonname, latname you use should match the variables you want to be able to access at the desired locations.

    Example usage
    -------------
    Select grid node of DataArray nearest to location (-96, 27). The DataArray `ds.temp` has coordinates `lon_rho` and `lat_rho`:
    >>> da = ds.temp
    >>> em.sel2d(da, lon_rho=-96, lat_rho=27)

    To additionally select from the time and depth dimensions, call as:
    >>> em.sel2d(da, lon_rho=-96, lat_rho=27, time='2022-07-19', s_rho=0.0)

    You may also input other keywords to pass onto `xarray.sel()`:
    >>> em.sel2d(da, lon_rho=-96, lat_rho=27, s_rho=0.0, method='nearest')
    """

    # assign input lon/lat coord names
    kwargs_iter = iter(kwargs)
    # first key is assumed longitude unless has "lat" in the name
    # second key is assumed latitude unless has "lon" in the name
    key1 = next(kwargs_iter)
    key2 = next(kwargs_iter)
    lonname, latname = key1, key2
    if ('lat' in key1) and ('lon' in key2):
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

    # 1D or 2D
    if lons.ndim == lats.ndim == 1:
        dims = 'loc'
    elif lons.ndim == lats.ndim == 2:
        dims = ('loc_y', 'loc_x')
    # else: Raise exception

    # create Dataset
    ds_to_find = xr.Dataset({
        'lat_to_find': (dims, lats),
        'lon_to_find': (dims, lons)
    })

    if var.xoak.index is None:
        var.xoak.set_index([latname, lonname], 'sklearn_geo_balltree')
    elif (latname, lonname) != var.xoak._index_coords:
        raise ValueError(f"Index has been built for grid with coords {var.xoak._index_coords} but coord names input are ({latname}, {lonname}).")

    # perform selection
    output = var.xoak.sel(
        {latname: ds_to_find.lat_to_find,
         lonname: ds_to_find.lon_to_find}
    )

    return output.sel(**kwargs)


def sel2dcf(var, **kwargs):
    """Find nearest value(s) on 2D horizontal grid using cf-xarray names.

    Use "longitude" and "latitude" for those coordinate names.

    You can input a combination of variable names and cf-xarray names for time and z dimensions.

    See `sel2d` for full docs. This wraps that function but will use cf-xarray standard names.

    Example usage
    -------------
    Select grid node of DataArray nearest to location (-96, 27):
    >>> da = ds.temp
    >>> em.sel2d(da, longitude=-96, latitude=27)

    To additionally select from the time and depth dimensions, call as:
    >>> em.sel2d(da, longitude=-96, latitude=27, T='2022-07-19', Z=0.0)

    You may also input other keywords to pass onto `xarray.sel()`:
    >>> em.sel2d(da, longitude=-96, latitude=27, Z=0.0, method='nearest')
    """

    lons, lats = kwargs['longitude'], kwargs['latitude']

    # use cf-xarray to get lon/lat key names
    try:
        latname = var.cf['latitude'].name
        lonname = var.cf['longitude'].name
    except KeyError:
        print('cf-xarray cannot determine variable name for longitude and latitude. Instead, use `sel2d()` and input the coordinate names specifically.')

    kwargs.pop('longitude')
    kwargs.pop('latitude')

    # need to maintain order
    new_kwargs = {lonname: lons, latname: lats}

    if 'T' in kwargs:
        tname = var.cf['T'].name
        new_kwargs[tname] = kwargs['T']
        kwargs.pop('T')
    if 'Z' in kwargs:
        zname = var.cf['Z'].name
        new_kwargs[zname] = kwargs['Z']
        kwargs.pop('Z')

    new_kwargs.update(kwargs)

    return sel2d(var, **new_kwargs)
