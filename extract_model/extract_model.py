"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""

import cf_xarray
import numpy as np
import xarray as xr
import xesmf as xe


def select(
    ds,
    longitude=None,
    latitude=None,
    varname=None,
    T=None,
    Z=None,
    iT=None,
    iZ=None,
    extrap=False,
    extrap_val=None,
    locstream=False,
):
    """Extract output from ds at location(s).

    Parameters
    ----------
    ds: Dataset
        Property to take gradients of.
    longitude, latitude: int, float, list, array (1D or 2D), DataArray, optional
        longitude(s), latitude(s) at which to return model output.
        Package `xESMF` will be used to interpolate with "bilinear" to
        these horizontal locations.
    varname: string, optional
        Name of variable in ds to interpolate.
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


    Returns
    -------
    DataArray of interpolated and/or selected values from ds.

    Example
    -------

    Select a single grid point.

    >>> longitude = 100
    >>> latitude = 10
    >>> iZ = 0
    >>> iT = 0
    >>> varname = 'u'
    >>> kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iT=iT, iZ=iZ, varname=varname)
    >>> dr = em.select(**kwargs)
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

    if varname is not None:

        dr = ds.cf[varname]
    else:
        dr = ds

    if (not extrap) and ((longitude is not None) and (latitude is not None)):
        assertion = "the input longitude range is outside the model domain"
        assert (longitude.min() >= dr.cf["longitude"].min()) and (
            longitude.max() <= dr.cf["longitude"].max()
        ), assertion
        assertion = "the input latitude range is outside the model domain"
        assert (latitude.min() >= dr.cf["latitude"].min()) and (
            latitude.max() <= dr.cf["latitude"].max()
        ), assertion

    ## Horizontal interpolation ##

    # grid of lon/lat to interpolate to, with desired ending attributes
    if (longitude is not None) and (latitude is not None):
        if latitude.ndim == 1:
            ds_out = xr.Dataset(
                {
                    "lat": (
                        ["lat"],
                        latitude,
                        dict(axis="Y", units="degrees_north", standard_name="latitude"),
                    ),
                    "lon": (
                        ["lon"],
                        longitude,
                        dict(axis="X", units="degrees_east", standard_name="longitude"),
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

        # set up regridder, which would work for multiple interpolations if desired
        regridder = xe.Regridder(
            dr, ds_out, "bilinear", extrap_method=extrap_method, locstream_out=locstream
        )

        # do regridding
        dr = regridder(dr, keep_attrs=True)

    ## Time and depth interpolation or iselection ##
    if iZ is not None:
        with xr.set_options(keep_attrs=True):
            dr = dr.cf.isel(Z=iZ)

    elif Z is not None:
        with xr.set_options(keep_attrs=True):
            dr = dr.cf.interp(Z=Z)

    if iT is not None:
        with xr.set_options(keep_attrs=True):
            dr = dr.cf.isel(T=iT)

    elif T is not None:
        with xr.set_options(keep_attrs=True):
            dr = dr.cf.interp(T=T)

    if extrap_val is not None:
        # returns 0 outside the domain by default. Assumes that no other values are exactly 0
        # and replaces all 0's with extrap_val if chosen.
        dr = dr.where(dr != 0, extrap_val)

    return dr
