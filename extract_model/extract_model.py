"""
Main file for this code. The main code is in `select`, and the rest is to help with variable name management.
"""
import numbers
import warnings

import cf_xarray  # noqa: F401
import numpy as np
import numpy.typing as npt
import pyinterp
import pyinterp.backends.xarray
import xarray as xr

try:
    import xesmf as xe
    XESMF = True
except ImportError:
    warnings.warn("sXESMF not found. Interpolation will be performed using pyinterp.")
    XESMF = False


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
    interp_lib="xesmf"
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

    # Horizontal interpolation
    ds_out = None
    if (longitude is not None) and (latitude is not None):
        ds_out = make_output_ds(longitude, latitude)

    if XESMF and interp_lib == "xesmf":
        da = _xesmf_interp(da, ds_out, T=T, Z=Z, iT=iT, iZ=iZ, extrap_method=extrap_method, extrap_val=extrap_val, locstream=locstream)
    elif interp_lib == "pyinterp":
        da = _pyinterp_interp(da, ds_out, T=T, Z=Z, iT=iT, iZ=iZ, extrap=extrap)
    else:
        raise ValueError(f"{interp_lib} interpolation not supported")

    return da


def _xesmf_interp(
    da,
    da_out=None,
    T=None,
    Z=None,
    iT=None,
    iZ=None,
    extrap_method='nearest_s2d',
    extrap_val=None,
    locstream=False,
):
    if da_out is not None:
        # set up regridder, which would work for multiple interpolations if desired
        regridder = xe.Regridder(
            da, da_out, "bilinear", extrap_method=extrap_method, locstream_out=locstream
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
    da,
    da_out=None,
    T=None,
    Z=None,
    iT=None,
    iZ=None,
    extrap=None,
):
    warnings.warn("extrap_method and locstream_out not supported for pyinterp")

    if extrap is not None:
        bounds_error = extrap
    else:
        bounds_error = False

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

    if da_out is not None:
        # Prepare points for interpolation
        # - Need a DataArray
        if type(da) == xr.Dataset:
            var_name = list(da.data_vars)[0]
            da = da[var_name]
        else:
            var_name = da.name

        # Add misssing coordinates to da_out
        if len(da_out.lon.shape) == 2:
            xy_dataset = xr.Dataset(data_vars={'X': np.arange(da_out.dims['X']), 'Y': np.arange(da_out.dims['Y'])})
            da_out = da_out.merge(xy_dataset)

        # Identify singular dimensions for time and depth
        def _is_singular_parameter(da, coordinate, vars):
            # First check if extraction parameters will render singular dimensions
            for v in vars:
                if v is not None:
                    if isinstance(v, list) and len(v) == 0:
                        return True
                    elif isinstance(v, numbers.Number):
                        return True

            # Then check if there are singular dimensions in the data array
            if coordinate in da.cf.coordinates:
                coordinate_name = da.cf.coordinates[coordinate][0]
                if da[coordinate_name].data.size == 1:
                    return True

            return False
        time_singular = _is_singular_parameter(da, 'time', [T, iT])
        vertical_singular = _is_singular_parameter(da, 'vertical', [Z, iZ])

        # Perform interpolation with details depending on dimensionality of data
        ndims = 0
        if 'longitude' in da.cf.coordinates:
            ndims += 1
        if 'latitude' in da.cf.coordinates:
            ndims += 1
        if 'vertical' in da.cf.coordinates and not vertical_singular:
            ndims += 1
        if 'time' in da.cf.coordinates and not time_singular:
            ndims += 1

        lat_var = da.cf.coordinates['latitude'][0]
        lon_var = da.cf.coordinates['longitude'][0]
        if 'time' in da.cf.coordinates:
            time_var = da.cf.coordinates['time'][0]
        else:
            time_var = None
        if 'vertical' in da.cf.coordinates:
            vertical_var = da.cf.coordinates['vertical'][0]
        else:
            vertical_var = None
        if ndims == 2:
            # Subset data
            subset_da = da
            if time_var:
                if time_var in subset_da.coords and time_var in subset_da.dims:
                    subset_da = subset_da.isel({time_var: 0})

            if vertical_var:
                if vertical_var in subset_da.coords and vertical_var in subset_da.dims:
                    subset_da = subset_da.isel({vertical_var: 0})

            # Interpolate
            try:
                mx, my = np.meshgrid(
                    da_out.lon.values,
                    da_out.lat.values,
                    indexing="ij"
                )
                grid = pyinterp.backends.xarray.Grid2D(subset_da)
                interped = grid.bivariate(
                    coords={
                        lon_var: mx.ravel(),
                        lat_var: my.ravel()
                    },
                    bounds_error=bounds_error
                ).reshape(mx.shape)
                # Transpose from x,y to y,x
                interped = interped.T
                regrid_method = 'bilinear'
            except ValueError:
                # Need to manually create grid when lon, lat are 2D (curvilinear or unstructured)
                grid = pyinterp.RTree()
                grid.packing(
                    np.vstack((subset_da[lon_var].data.ravel(), subset_da[lat_var].data.ravel())).T,
                    subset_da.data.ravel(),
                )
                if len(da_out.lon.shape) == 2:
                    mx = da_out.lon.values
                    my = da_out.lat.values
                else:
                    mx, my = np.meshgrid(
                        da_out.lon.values,
                        da_out.lat.values,
                        indexing="ij"
                    )
                idw, _ = grid.inverse_distance_weighting(
                    np.vstack((mx.ravel(), my.ravel())).T,
                    within=extrap,
                    k=5,
                )
                interped = idw.reshape(mx.shape)
                regrid_method = 'IDW'

            # Package as DataArray
            if len(da_out.lon) == 1:
                lons = da_out.lon.isel({'lon': 0})
            else:
                lons = da_out.lon
            if len(da_out.lat) == 1:
                lats = da_out.lat.isel({'lat': 0})
            else:
                lats = da_out.lat

            coords = {
                'lon': lons,
                'lat': lats
            }
            # Handle curvilinear lon/lat coords
            if len(lons.shape) == 2:
                for dim in lons.dims:
                    coords[dim] = lons[dim]
            if 'time' in da.cf.coordinates:
                coords['time'] = da[time_var]
            if 'vertical' in da.cf.coordinates:
                coords['vertical'] = da[vertical_var]

            # Handle missing dims from interpolation
            missing_subset_dims = []
            for subset_dim in subset_da.dims:
                if subset_dim not in [da.cf.coordinates['longitude'][0], da.cf.coordinates['latitude'][0]]:
                    missing_subset_dims.append(subset_dim)

            output_dims = []
            for orig_dim in da.dims:
                # Handle original x, y to lon, lat
                # Also, do not add lon and lat if they are scalars
                if orig_dim == 'xi_rho' and len(da_out.lon) > 1:
                    output_dims.append('X')
                elif orig_dim == 'xi_rho' and len(da_out.lon) == 1:
                    interped = np.squeeze(interped, axis=0)
                    continue
                elif orig_dim == 'eta_rho' and len(da_out.lat) > 1:
                    output_dims.append('Y')
                elif orig_dim == 'eta_rho' and len(da_out.lat) == 1:
                    interped = np.squeeze(interped, axis=0)
                    continue
                elif orig_dim == da.cf.coordinates['longitude'][0] and len(da_out.lon) > 1:
                    output_dims.append('lon')
                elif orig_dim == da.cf.coordinates['longitude'][0] and len(da_out.lon) == 1:
                    interped = np.squeeze(interped, axis=0)
                    continue
                elif orig_dim == da.cf.coordinates['latitude'][0] and len(da_out.lat) > 1:
                    output_dims.append('lat')
                elif orig_dim == da.cf.coordinates['latitude'][0] and len(da_out.lat) == 1:
                    interped = np.squeeze(interped, axis=0)
                    continue
                else:
                    output_dims.append(orig_dim)

                    if orig_dim not in missing_subset_dims:
                        interped = interped[np.newaxis, ...]
            da = xr.DataArray(
                interped,
                coords=coords,
                dims=output_dims,
                attrs=da.attrs | {'regrid_method': regrid_method}
            )
        elif ndims == 3:
            # Subset data
            subset_da = da
            time_da = subset_da[time_var]
            vertical_da = subset_da[vertical_var]
            if time_singular:
                if iT is not None:
                    subset_da = subset_da.isel({time_var: iT})
                    time_da = time_da.isel({time_var: iT})
                else:
                    subset_da = subset_da.sel({time_var: T})
                    time_da = time_da.sel({time_var: T})
            if vertical_singular:
                if iZ is not None:
                    subset_da = subset_da.isel({vertical_var: iZ})
                    vertical_da = vertical_da.isel({time_var: iT})
                else:
                    subset_da = subset_da.sel({vertical_var: Z})
                    vertical_da = vertical_da.sel({time_var: Z})

            grid = pyinterp.backends.xarray.Grid3D(subset_da)
            mx, my, mz = np.meshgrid(
                da_out.lon.values,
                da_out.lat.values,
                da.cf.coords['time'].values,
                indexing="ij"
            )
            interped = grid.bicubic(
                coords={"lon": mx.ravel(), "lat": my.ravel(), "time": mz.ravel()},
                bounds_error=bounds_error
            ).reshape(mx.shape)
            coords = {
                "lat": da_out.lat,
                "lon": da_out.lon,
                "time": da.cf.coords["time"],
            }
            da = xr.Dataset(
                {var_name: (["lat", "lon", "time"], interped)},
                coords=coords,
                attrs=da.attrs,
            )
        elif ndims == 4:
            grid = pyinterp.backends.xarray.Grid4D(da)
            mx, my, mz, mu = np.meshgrid(
                da_out.lon.values,
                da_out.lat.values,
                da.cf.coords['time'].values,
                da.cf.coords['vertical'].values,
                indexing="ij"
            )
            interped = grid.bicubic(
                coords={"lon": mx.ravel(), "lat": my.ravel(), "time": mz.ravel(), "vertical": mu.ravel()},
                bounds_error=bounds_error
            ).reshape(mx.shape)
            coords = {
                "lat": da_out.lat,
                "lon": da_out.lon,
                "time": da.cf.coords["time"],
                "vertical": da.cf.coords["vertical"],
            }
            da = xr.Dataset(
                {var_name: (["lat", "lon", "time", "vertical"], interped)},
                coords=coords,
                attrs=da.attrs,
            )
        else:
            raise IndexError(f"{ndims}D interpolation not supported")

    return da


def make_output_ds(longitude: npt.ArrayLike, latitude: npt.ArrayLike) -> xr.Dataset:
    """
    Given desired interpolated longitude and latitude, return points as Dataset.
    """
    # Grid of lat/lon to interpolate to with desired ending attributes
    ds_out = None
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
        else:
            raise IndexError(f"{latitude.ndim}D latitude/longitude arrays not supported.")

    return ds_out
