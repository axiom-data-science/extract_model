"""
This is an accessor to xarray. It is basically a convenient way to
use the extract_model functions, and has bookkeeping in the
background where possible. No new functions are available only here;
this connects to functions in other files.
"""

import numpy as np
import xarray as xr

# from xgcm import grid as xgrid

# import xroms
import extract_model as em


xr.set_options(keep_attrs=True)

# g = 9.81  # m/s^2


@xr.register_dataset_accessor("em")
class emDatasetAccessor:
    def __init__(self, ds):

        self.ds = ds

        # extra for getting coordinates but changes variables
        self._ds = ds.copy(deep=True)

#     @property
#     def sel2d(self):
#         """Calculate horizontal speed [m/s] from u and v components, on rho/rho grids.
#         Notes
#         -----
#         speed = np.sqrt(u^2 + v^2)
#         Uses 'extend' for horizontal boundary.
#         See `xroms.speed` for full docstring.
#         Example usage
#         -------------
#         >>> ds.xroms.speed
#         """

#         if "speed" not in self.ds:
#             var = xroms.speed(self.ds.u, self.ds.v, self.grid, hboundary="extend")
#             self.ds["speed"] = var
#         return self.ds.speed


@xr.register_dataarray_accessor("em")
class emDataArrayAccessor:
    def __init__(self, da):

        self.da = da
        self.argsel2d_map = {}
        self.regridder_map = {}
        
        # improve this logic over time
        if 'node' in da.dims:
            self.is_unstructured = True
        else:
            self.is_unstructured = False
        
        # assign DataArray dims with axis labels so cf_xarray will work
        # assume the dimensions are in [T,Z,Y,X] order
        axis_struc, axis_unstruc = {}, {}
        axis_struc[4] = ['T','Z','Y','X']
        axis_struc[3] = ['T','Y','X']
        axis_struc[2] = ['Y','X']
        axis_unstruc[3] = ['T', 'Z', 'X']  # unstructured
        axis_unstruc[2] = ['T', 'X']  # unstructured
        axis_unstruc[1] = ['X']  # unstructured
        
        if self.is_unstructured:
            axiss = axis_unstruc
        else:
            axiss = axis_struc

        for dim, axis in zip(da.dims, axiss[len(da.dims)]):
            # dim needs to be a coord to have attributes
            if dim not in da.coords:
                self.da[dim] = (
                    dim,
                    np.arange(da.sizes[dim]),
                    {"axis": axis},
                )
            else:
                self.da[dim].attrs["axis"] = axis
        

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

    def sel2d(self, lon0, lat0, **kwargs):# T=None, iT=None):
        """Find the value of the var at closest location to lon0,lat0.
        
        Inputs
        ------
        lon0: float, int
            Longitude of comparison point.
        lat0: float, int
            Latitude of comparison point.
        T: datetime-like string, list of datetime-like strings, optional
            Datetime or datetimes at which to return model output.
            `xarray`'s built-in 1D selection will be used to calculate.
            To select time in any way, use either this keyword argument
            or `iT`, but not both simultaneously.
        iT: int or list of ints, optional
            Index of time in time coordinate to select using `.isel`.
            To select time in any way, use either this keyword argument
            or `T`, but not both simultaneously.
            
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
            self.da, self.da.cf["longitude"], self.da.cf["latitude"], lon0, lat0, 
            inds=self.argsel2d_map[(lon0, lat0)], **kwargs)
#             iT=iT, T=T
#         )

    def interp2d(self, lons, lats, locstream=False, T=None, iT=None, Z=None, iZ=None, extrap=False, extrap_val=None):# which="pairs"):
        """Interpolate var to lons/lats positions.
        
        Wraps xESMF to perform proper horizontal interpolation on non-flat Earth.
        
        UPDATE TO MATCH select
        
        Inputs
        ------
        lons: list, ndarray
            Longitudes to interpolate to. Will be flattened upon input.
        lats: list, ndarray
            Latitudes to interpolate to. Will be flattened upon input.
        which: str, optional
            Which type of interpolation to do:
            * "pairs": lons/lats as unstructured coordinate pairs
              (in xESMF language, LocStream).
            * "grid": 2D array of points with 1 dimension the lons and
              the other dimension the lats.
        T: datetime-like string, list of datetime-like strings, optional
            Datetime or datetimes at which to return model output.
            `xarray`'s built-in 1D selection will be used to calculate.
            To select time in any way, use either this keyword argument
            or `iT`, but not both simultaneously.
        iT: int or list of ints, optional
            Index of time in time coordinate to select using `.isel`.
            To select time in any way, use either this keyword argument
            or `T`, but not both simultaneously.
        UPDATE
              
        Returns
        -------
        DataArray of var interpolated to lons/lats. Dimensionality will be the
        same as var except the Y and X dimensions will be 1 dimension called
        "locations" that lons.size if which=='pairs', or 2 dimensions called
        "lat" and "lon" if which=='grid' that are of lats.size and lons.size,
        respectively.
        
        Notes
        -----
        var cannot have chunks in the Y or X dimensions.
        cf-xarray should still be usable after calling this function.
        
        Example usage
        -------------
        To return 1D pairs of points, in this case 3 points:
        >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='pairs')
        To return 2D pairs of points, in this case a 3x3 array of points:
        >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='grid')
        """
        
        import hashlib

        if isinstance(lons, (xr.DataArray, xr.Dataset)):
            lons, lats = lons.values, lats.values
        if not isinstance(lons, (list, np.ndarray)):
            lons, lats = np.array([lons]), np.array([lats])
#             lons, lats = [lons], [lats]
        if not isinstance(lons, np.ndarray):
            lons, lats = np.array(lons), np.array(lats)
        
        coords = sorted(list(zip(*(lons.flatten(), lats.flatten()))))
        hashenc = hashlib.sha256(str(coords).encode()).hexdigest()
        
        # first see if already know the mapping
        if hashenc in self.regridder_map:
            regridder = self.regridder_map[hashenc]
            da, _ = em.select(self.da, longitude=lons, latitude=lats, locstream=locstream, regridder=regridder, iT=iT, T=T, iZ=iZ, Z=Z, extrap=extrap, extrap_val=extrap_val)# which=which)
        else:
            da, regridder = em.select(self.da, longitude=lons, latitude=lats, locstream=locstream, iT=iT, T=T, iZ=iZ, Z=Z, extrap=extrap, extrap_val=extrap_val)# which=which)
            self.regridder_map[hashenc] = regridder

        return da

    
    def subset(self, bbox, drop=True):
        """COPY FROM OTHER"""
        
        return em.subset(self.da, bbox, drop=drop)
        