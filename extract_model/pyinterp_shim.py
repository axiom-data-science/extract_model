"""
Temporary interface for using pyinterp.
"""
import numbers
import warnings
from typing import Tuple

import numpy as np
import xarray as xr

try:
    import pyinterp
    import pyinterp.backends.xarray
    import pyinterp.fill
except ImportError:
    warnings.warn("pyinterp not installed. Interpolation will be performed using xESMF.")


class PyInterpShim:

    def __call__(
        self,
        da,
        da_out=None,
        T=None,
        Z=None,
        iT=None,
        iZ=None,
        extrap=None,
        locstream=False,
    ):
        warnings.warn("extrap_method not supported for pyinterp.")

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

        # Requires horizontal interpolation
        if da_out is not None:
            # interpolate to the output grid
            # then package appropriately
            subset_da, interped_array, interp_method = self._interp(da, da_out, T, Z, iT, iZ, bounds_error)
            if locstream:
                da = self._package_locstream(da, da_out, subset_da, interped_array, T, Z, iT, iZ, interp_method)
            else:
                da = self._package_grid(da, da_out, subset_da, interped_array, T, Z, iT, iZ, interp_method)

        return da

    def _interp(
        self,
        da,
        da_out,
        T=None,
        Z=None,
        iT=None,
        iZ=None,
        bounds_error=None
    ) -> Tuple[xr.DataArray, np.ndarray, str]:
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
        regrid_method = 'bilinear'

        subset_da = da
        if ndims == 2:
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
            except ValueError:
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
                    within=bounds_error,
                    k=5,
                )
                interped = idw.reshape(mx.shape)
                regrid_method = 'IDW'

        elif ndims == 3:
            if time_var:
                time_da = subset_da[time_var]
            if vertical_var:
                vertical_da = subset_da[vertical_var]

            if time_singular:
                if iT is not None:
                    subset_da = subset_da.isel({time_var: iT})
                    time_da = time_da.isel({time_var: iT})
                elif T is not None:
                    subset_da = subset_da.sel({time_var: T})
                    time_da = time_da.sel({time_var: T})
            if vertical_singular:
                if iZ is not None:
                    subset_da = subset_da.isel({vertical_var: iZ})
                    vertical_da = vertical_da.isel({time_var: iT})
                elif Z is not None:
                    subset_da = subset_da.sel({vertical_var: Z})
                    vertical_da = vertical_da.sel({time_var: Z})

            # Regular grid
            try:
                mx, my, mz = np.meshgrid(
                    da_out.lon.values,
                    da_out.lat.values,
                    da.cf.coords['time'].values,
                    indexing="ij"
                )

                # Fill NaNs using Loess
                grid = pyinterp.backends.xarray.Grid3D(subset_da)
                filled = pyinterp.fill.loess(grid, nx=5, ny=5)
                grid = pyinterp.Grid3D(grid.x, grid.y, grid.z, filled)
                interped = pyinterp.bicubic(
                    grid,
                    x=mx.ravel(),
                    y=my.ravel(),
                    z=mz.ravel(),
                    bounds_error=bounds_error
                ).reshape(mx.shape)
            # Curviliear or unstructured
            except ValueError:
                # Need to manually create grid when lon, lat are 2D (curvilinear or unstructured)
                trailing_dim = subset_da.shape[0]

                grid = pyinterp.RTree()
                grid.packing(
                    np.vstack((subset_da[lon_var].data.ravel(), subset_da[lat_var].data.ravel())).T,
                    subset_da.data.ravel().reshape(-1, trailing_dim),
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
                    within=bounds_error,
                    k=5,
                )
                interped = idw.reshape(mx.shape)
                regrid_method = 'IDW'

        elif ndims == 4:
            mx, my, mz, mu = np.meshgrid(
                da_out.lon.values,
                da_out.lat.values,
                da.cf.coords['time'].values,
                da.cf.coords['vertical'].values,
                indexing="ij"
            )
            # Fill NaNs using Loess
            grid = pyinterp.backends.xarray.Grid4D(da)
            filled = pyinterp.fill.loess(grid, nx=3, ny=3)
            grid = pyinterp.Grid4D(grid.x, grid.y, grid.z, grid.u, filled)
            interped = pyinterp.bicubic(
                grid,
                x=mx.ravel(),
                y=mx.ravel(),
                z=mz.ravel(),
                u=mu.ravel(),
                bounds_error=bounds_error
            ).reshape(mx.shape)
        else:
            raise IndexError(f"{ndims}D interpolation not supported")

        return subset_da, interped, regrid_method

    def _package_locstream(
        self,
        da,
        da_out,
        subset_da,
        interped,
        T=None,
        Z=None,
        iT=None,
        iZ=None,
        regrid_method=None
    ):
        # Prepare points for interpolation
        # - Need a DataArray
        if type(da) == xr.Dataset:
            var_name = list(da.data_vars)[0]
            da = da[var_name]
        else:
            var_name = da.name

        # Locstream will have dim pt for the number of points
        # - Change dims from lon/lat to pts
        lat_var = da_out.cf.coordinates['latitude'][0]
        lon_var = da_out.cf.coordinates['longitude'][0]
        da_out = da_out.rename_dims(
            {
                lat_var: 'pts',
                lon_var: 'pts',
            }
        )

        # Add coordinates from the original data
        coords = da_out.coords
        if 'time' in da.cf.coordinates:
            time_var = da.cf.coordinates['time'][0]
        else:
            time_var = None
        if 'vertical' in da.cf.coordinates:
            vertical_var = da.cf.coordinates['vertical'][0]
        else:
            vertical_var = None

        if 'time' in da.cf.coordinates:
            coords['time'] = subset_da[time_var]
        if 'vertical' in da.cf.coordinates:
            coords['vertical'] = subset_da[vertical_var]

        # Add interpolated data
        # If a single point, reshape to len(pts, 1)
        if da_out[lat_var].shape == (1,):
            interped = np.squeeze(interped)[:, np.newaxis]
            # Also need to swap the dims to match
            dims = [dim for dim in da_out.dims][::-1]
        # Else, it's probably a grid and the diagonal needs to be extracted
        # - This is a workaround for a bad implementation in _interp which interpolates
        #   a whole grid instead of just a set of points.
        else:
            interped = np.diagonal(interped)
            dims = da_out.dims

        return xr.DataArray(
            interped,
            coords=coords,
            dims=dims,
            attrs={**da.attrs, **{'regrid_method': regrid_method}}
        )

    def _package_grid(
        self,
        da,
        da_out,
        subset_da,
        interped,
        T=None,
        Z=None,
        iT=None,
        iZ=None,
        regrid_method=None
    ):
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

        if 'time' in da.cf.coordinates:
            time_var = da.cf.coordinates['time'][0]
        else:
            time_var = None
        if 'vertical' in da.cf.coordinates:
            vertical_var = da.cf.coordinates['vertical'][0]
        else:
            vertical_var = None
        if ndims == 2:
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
                attrs={**da.attrs, **{'regrid_method': regrid_method}}
            )
        elif ndims == 3:
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
