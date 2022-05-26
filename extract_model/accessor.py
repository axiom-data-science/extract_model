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


@xr.register_dataarray_accessor("em")
class emDataArrayAccessor:
    """DataArray accessor."""

    def __init__(self, da):
        "Initialize."

        self.da = da
        self.argsel2d_map = {}
        self.regridder_map = {}

    #         # improve this logic over time
    #         if 'node' in da.dims:
    #             self.is_unstructured = True
    #         else:
    #             self.is_unstructured = False

    def argsel2d(self, lon0, lat0):
        """Find the indices of coordinate pair closest to another point.

        Inputs
        ------
        lon0: float, int
            Longitude of comparison point.
        lat0: float, int
            Latitude of comparison point.

        Returns
        -------
        Indices in eta, xi of closest location to lon0, lat0.

        Notes
        -----
        This function uses Great Circle distance to calculate distances assuming
        longitudes and latitudes as point coordinates. Uses cartopy function
        `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

        Example usage
        -------------
        >>> ds.temp.xroms.argsel2d(-96, 27)
        """

        # first see if already know the mapping
        if (lon0, lat0) in self.argsel2d_map:
            return self.argsel2d_map[(lon0, lat0)]

        # save location mapping in case requested again:
        else:
            inds = em.argsel2d(
                self.da.cf["longitude"], self.da.cf["latitude"], lon0, lat0
            )
            self.argsel2d_map[(lon0, lat0)] = tuple(inds)

            return self.argsel2d_map[(lon0, lat0)]

    def sel2d(self, lon0, lat0, **kwargs):
        """Find the value of the var at closest location to lon0,lat0.

        Inputs
        ------
        lon0: float, int
            Longitude of comparison point.
        lat0: float, int
            Latitude of comparison point.
        kwargs: see `em.sel2d()` for full docs.

        Returns
        -------
        DataArray value(s) of closest location to lon0/lat0.

        Notes
        -----
        This function uses Great Circle distance to calculate distances assuming
        longitudes and latitudes as point coordinates. Uses cartopy function
        `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html
        This wraps `argsel2d`.

        Example usage
        -------------
        >>> ds.temp.em.sel2d(-96, 27)
        """

        self.argsel2d(lon0, lat0)

        return em.sel2d(
            self.da,
            self.da.cf["longitude"],
            self.da.cf["latitude"],
            lon0,
            lat0,
            inds=self.argsel2d_map[(lon0, lat0)],
            **kwargs
        )

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

        Inputs
        ------
        lons: list, ndarray, optional
            Longitudes to interpolate to. Will be flattened upon input.
        lats: list, ndarray, optional
            Latitudes to interpolate to. Will be flattened upon input.
        Full docs are available at `em.interp2d()`.

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

        if hashenc in self.regridder_map:
            regridder = self.regridder_map[hashenc]
            da, _ = em.select(
                self.da,
                longitude=lons,
                latitude=lats,
                locstream=locstream,
                weights=weights,
                regridder=regridder,
                iT=iT,
                T=T,
                iZ=iZ,
                Z=Z,
                extrap=extrap,
                extrap_val=extrap_val,
            )
        else:
            da, regridder = em.select(
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
            self.regridder_map[hashenc] = regridder

        return da

    def sub_bbox(self, bbox, drop=True, dask_array_chunks=True):
        """Subset DataArray in space defined by bbox.

        See full docs at `em.sub_bbox()`.
        """

        return em.sub_bbox(
            self.da, bbox, drop=drop, dask_array_chunks=dask_array_chunks
        )
