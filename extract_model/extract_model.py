"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""
import warnings
from numbers import Number
from typing import Optional, Union

import cf_xarray  # noqa: F401
import numpy as np
import numpy.typing as npt
import xarray as xr

try:
    import xesmf as xe

    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    warnings.warn("xESMF not found. Interpolation will be performed using pyinterp.")

try:
    from .pyinterp_shim import PyInterpShim
except ImportError:
    if XESMF_AVAILABLE:
        warnings.warn(
            "PyInterp not found. Interpolation will be performed using xESMF."
        )
    else:
        raise ModuleNotFoundError(
            "Neither PyInterp nor xESMF are available. Please install either package."
        )


def select(
    da: xr.DataArray,
    longitude: Optional[
        Union[Number, list[Number], npt.ArrayLike, xr.DataArray]
    ] = None,
    latitude: Optional[Union[Number, list[Number], npt.ArrayLike, xr.DataArray]] = None,
    T: Optional[Union[str, list[str]]] = None,
    Z: Optional[Union[Number, list[Number]]] = None,
    iT: Optional[Union[int, list[int]]] = None,
    iZ: Optional[Union[int, list[int]]] = None,
    extrap: bool = False,
    extrap_val: Optional[Number] = None,
    locstream: bool = False,
    interp_lib: str = "xesmf",
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
        or as `extra_value` if input.
    locstream: boolean, optional
        Which type of interpolation to do:

        * False: 2D array of points with 1 dimension the lons and the other dimension the lats.
        * True: lons/lats as unstructured coordinate pairs (in xESMF language, LocStream).


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
        output_grid = True
    else:
        output_grid = False

    # Horizontal interpolation
    # Verify interpolated points in domain if not extrapolating.
    if output_grid and not extrap:
        if (
            longitude.min() < da.cf["longitude"].min()
            or longitude.max() > da.cf["longitude"].max()
        ):
            raise ValueError(
                "Longitude outside of available domain."
                "Use extrap=True to extrapolate."
            )
        if (
            latitude.min() < da.cf["latitude"].min()
            or latitude.max() > da.cf["latitude"].max()
        ):
            raise ValueError(
                "Latitude outside of available domain."
                "Use extrap=True to extrapolate."
            )

    # Create output grid as Dataset.
    if output_grid:
        ds_out = make_output_ds(longitude, latitude)
    else:
        ds_out = None

    # If extrapolating, define method
    if extrap:
        extrap_method = "nearest_s2d"
    else:
        extrap_method = None

    # Perform interpolation
    if interp_lib == "xesmf" and XESMF_AVAILABLE:
        da = _xesmf_interp(
            da,
            ds_out,
            T=T,
            Z=Z,
            iT=iT,
            iZ=iZ,
            extrap_method=extrap_method,
            extrap_val=extrap_val,
            locstream=locstream,
        )
    elif interp_lib == "pyinterp" or not XESMF_AVAILABLE:
        da = _pyinterp_interp(
            da,
            ds_out,
            T=T,
            Z=Z,
            iT=iT,
            iZ=iZ,
            extrap_method=extrap_method,
            extrap_val=extrap_val,
            locstream=locstream,
        )
    else:
        raise ValueError(f"{interp_lib} interpolation not supported")

    return da


def _xesmf_interp(
    da: xr.DataArray,
    ds_out: Optional[xr.Dataset] = None,
    T: Optional[Union[str, list[str]]] = None,
    Z: Optional[Union[Number, list[Number]]] = None,
    iT: Optional[Union[int, list[int]]] = None,
    iZ: Optional[Union[int, list[int]]] = None,
    extrap_method: Optional[str] = None,
    extrap_val: Optional[Number] = None,
    locstream: bool = False,
) -> xr.DataArray:
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
    extrap: bool, optional
    extrap_val: int, float, optional
    locstream: boolean, optional

    Returns
    -------
    DataArray of interpolated and/or selected values from da.
    """
    if ds_out is not None:
        # set up regridder, which would work for multiple interpolations if desired
        regridder = xe.Regridder(
            da, ds_out, "bilinear", extrap_method=extrap_method, locstream_out=locstream
        )
        da = regridder(da, keep_attrs=True)

    # Time and depth interpolation or iselection
    with xr.set_options(keep_attrs=True):
        if iZ is not None:
            da = da.cf.isel(Z=iZ)
        elif Z is not None:
            da = da.cf.interp(Z=Z)

        if iT is not None:
            da = da.cf.isel(T=iT)
        elif T is not None:
            da = da.cf.interp(T=T)

    if extrap_val is not None:
        # returns 0 outside the domain by default. Assumes that no other values are exactly 0
        # and replaces all 0's with extrap_val if chosen.
        da = da.where(da != 0, extrap_val)

    return da


def _pyinterp_interp(
    da: xr.DataArray,
    ds_out: Optional[xr.Dataset] = None,
    T: Optional[Union[str, list[str]]] = None,
    Z: Optional[Union[Number, list[Number]]] = None,
    iT: Optional[Union[int, list[int]]] = None,
    iZ: Optional[Union[int, list[int]]] = None,
    extrap_method: Optional[str] = None,
    extrap_val: Optional[Number] = None,
    locstream: bool = False,
):
    """Interpolate input DataArray to output DataArray using PyInterp.

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
    extrap: bool, optional
    extrap_val: int, float, optional
    locstream: boolean, optional

    Returns
    -------
    DataArray of interpolated and/or selected values from da.
    """

    # Loess based extrapolation will be used if required.
    if extrap_method is not None:
        extrap = True
    else:
        extrap = False

    interpretor = PyInterpShim()
    da = interpretor(
        da, ds_out, T=T, Z=Z, iT=iT, iZ=iZ, extrap=extrap, locstream=locstream
    )

    return da


def make_output_ds(longitude: npt.ArrayLike, latitude: npt.ArrayLike) -> xr.Dataset:
    """
    Given desired interpolated longitude and latitude, return points as Dataset.
    """
    # Grid of lat/lon to interpolate to with desired ending attributes
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
    else:
        raise IndexError(f"{latitude.ndim}D latitude/longitude arrays not supported.")

    return ds_out
