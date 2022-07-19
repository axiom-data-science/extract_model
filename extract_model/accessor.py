"""
This is an accessor to xarray. It is basically a convenient way to
use the extract_model functions, and has bookkeeping in the
background where possible. No new functions are available only here;
this connects to functions in other files.
"""

import numpy as np
import xarray as xr

import extract_model as em


xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("em")
class emDatasetAccessor:
    """Dataset accessor."""

    def __init__(self, ds):
        """Initialize."""

        self.ds = ds

        # extra for getting coordinates but changes variables
        self._ds = ds.copy(deep=True)

    def sub_grid(self, bbox, drop=True):
        """Subset Dataset in space defined by bbox.

        Returns full set of grids that preserve structure.

        See full docs at `em.sub_grid()`.
        """

        return em.sub_grid(self.ds, bbox)

    def sub_bbox(self, bbox, drop=True, dask_array_chunks=True):
        """Subset DataArray in space defined by bbox.

        See full docs at `em.sub_bbox()`.
        """

        attrs = self.ds.attrs

        dss = []
        Vars = [
            Var for Var in self.ds.data_vars if "longitude" in self.ds[Var].cf.coords
        ]
        for Var in self.ds.data_vars:
            if Var in Vars:
                dss.append(
                    em.sub_bbox(
                        self.ds[Var],
                        bbox,
                        drop=drop,
                        dask_array_chunks=dask_array_chunks,
                    )
                )
            else:
                dss.append(self.ds[Var])

        ds_out = xr.merge(dss)

        ds_out.attrs = attrs

        return ds_out

    def filter(self, standard_names):
        """Filter Dataset to standard_names, keep imp variables too.

        See full docs at `em.utils.filter()`.
        """

        return em.filter(self.ds, standard_names)

    def sel2d(self, **kwargs):
        """Find nearest value(s) on horizontal grid.

        Can also pass through `xarray` `.sel` information for other dimensions. See `em.sel2d()` for full docs.
        """
        return em.sel2d(self.ds, **kwargs)

    def sel2dcf(self, **kwargs):
        """Find nearest value(s) on horizontal grid.

        Use `cf-xarray` nicknames for horizontal coordinates: 'longitude' and 'latitude'. Can also pass through `xarray` `.sel` information for other dimensions. See `em.sel2d()` for full docs.
        """
        return em.sel2dcf(self.ds, **kwargs)


@xr.register_dataarray_accessor("em")
class emDataArrayAccessor:
    """DataArray accessor."""

    def __init__(self, da):
        "Initialize."

        self.da = da
        # self.argsel2d_map = {}
        self.weights_map = {}

        # interp2d should be called `select` to be consistent with original
        # function, so making either work here
        self.select = self.interp2d

    #         # improve this logic over time
    #         if 'node' in da.dims:
    #             self.is_unstructured = True
    #         else:
    #             self.is_unstructured = False

    def sel2d(self, **kwargs):
        """Find nearest value(s) on 2D horizontal grid.

        Can also pass through `xarray` `.sel` information for other dimensions. See `em.sel2d()` for full docs.
        """
        return em.sel2d(self.da, **kwargs)

    def sel2dcf(self, **kwargs):
        """Find nearest value(s) on 2D horizontal grid.

        Use `cf-xarray` nicknames for horizontal coordinates: 'longitude' and 'latitude'. Can also pass through `xarray` `.sel` information for other dimensions. See `em.sel2d()` for full docs.
        """
        return em.sel2dcf(self.da, **kwargs)

    def interp2d(
        self,
        lons=None,
        lats=None,
        locstream=False,
        T=None,
        iT=None,
        Z=None,
        iZ=None,
        extrap=False,
        extrap_val=None,
        weights=None,
    ):
        """Interpolate var to lons/lats positions.

        Wraps xESMF to perform proper horizontal interpolation on non-flat Earth.
        This reuses the calculated weights behind the scenes if the same
        interpolation is requested. Does not work if xESMF is not installed.

        Inputs
        ------
        lons: list, ndarray, optional
            Longitudes to interpolate to. Will be flattened upon input.
        lats: list, ndarray, optional
            Latitudes to interpolate to. Will be flattened upon input.
        Full docs are available at `em.select()`.

        Returns
        -------
        DataArray of var interpolated to lons/lats. Dimensionality will be the
        same as var except the Y and X dimensions will be 1 dimension called
        "locations" that lons.size if `locstream=True`, or 2 dimensions called
        "lat" and "lon" if `locstream=False` that are of lats.size and lons.size,
        respectively.

        Notes
        -----
        var cannot have chunks in the Y or X dimensions.
        cf-xarray should still be usable after calling this function.

        Example usage
        -------------
        To return 1D pairs of points, in this case 3 points:
        >>> da.em.interp2d([-96, -97, -96.5], [26.5, 27, 26.5], locstream=True)
        To return 2D pairs of points, in this case a 3x3 array of points:
        >>> da.em.interp2d([-96, -97, -96.5], [26.5, 27, 26.5], locstream=False)
        """

        import hashlib

        if isinstance(lons, (xr.DataArray, xr.Dataset)):
            lons, lats = lons.values, lats.values
        if not isinstance(lons, (list, np.ndarray)):
            lons, lats = np.array([lons]), np.array([lats])
        if not isinstance(lons, np.ndarray):
            lons, lats = np.array(lons), np.array(lats)

        if locstream:
            coords = sorted(list(zip(*(lons.flatten(), lats.flatten()))))
        else:
            # coords should be gridded
            Lon, Lat = np.meshgrid(lons.flatten(), lats.flatten())
            coords = sorted(list(zip(*(Lon.flatten(), Lat.flatten()))))

        hashenc = hashlib.sha256(str(coords).encode()).hexdigest()

        if hashenc in self.weights_map:
            weights = self.weights_map[hashenc]
            da, _ = em.select(
                self.da,
                longitude=lons,
                latitude=lats,
                locstream=locstream,
                weights=weights,
                iT=iT,
                T=T,
                iZ=iZ,
                Z=Z,
                extrap=extrap,
                extrap_val=extrap_val,
            )
        else:
            da, weights = em.select(
                self.da,
                longitude=lons,
                latitude=lats,
                locstream=locstream,
                weights=weights,
                iT=iT,
                T=T,
                iZ=iZ,
                Z=Z,
                extrap=extrap,
                extrap_val=extrap_val,
            )
            self.weights_map[hashenc] = weights

        return da

    def sub_bbox(self, bbox, drop=True, dask_array_chunks=True):
        """Subset DataArray in space defined by bbox.

        See full docs at `em.sub_bbox()`.
        """

        return em.sub_bbox(
            self.da, bbox, drop=drop, dask_array_chunks=dask_array_chunks
        )
