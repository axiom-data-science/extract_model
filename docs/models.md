---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3.8.10 ('extract_model_docs')
  language: python
  name: python3
---

# Generically access model output

```{code-cell} ipython3
:tags: []

import cf_xarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xcmocean
import cmocean.cm as cmo
import extract_model as em
```

## ROMS

```{code-cell} ipython3
# open an example dataset from xarray's tutorials
ds = xr.tutorial.open_dataset('ROMS_example.nc', chunks={'ocean_time': 1})
# normally could run the `preprocess` code as part of reading in the dataset
# but with the tutorial model output, run it separately:
ds = em.preprocess(ds)
ds
```

Note that the preprocessing code sets up a ROMS dataset so that it can be used with `cf-xarray`. For example, axis and coordinate variables have been identified:

```{code-cell} ipython3
:tags: []

ds.cf
```

Variable to use, by standard_name:

```{code-cell} ipython3
zeta = 'sea_surface_elevation'
salt = 'sea_water_practical_salinity'
```

### Subset numerical domain

Use `.em.sub_grid()` to narrow the model area down using a bounding box on a Dataset which respects the horizontal structure of multiple grids. Currently only is relevant for ROMS models but will run on any ROMS model or models with a single longitude/latitude set of coordinates.

Resulting area of model will not be exactly the bounding box if the domain is curvilinear.

```{code-cell} ipython3
ds_sub = ds.em.sub_grid([-92, 27, -90, 29])
ds_sub.cf[zeta].cf.isel(T=0).cf.plot(x='longitude', y='latitude')
```

Note that this is an unusual ROMS Dataset because it has only one horizontal grid.

```{code-cell} ipython3
ds_sub
```

### Subset to a horizontal box

Use `.em.sub_bbox()` to narrow the model area down using a bounding box on either a Dataset or DataArray. There is no expectation of multiple horizontal grids having the "correct" relationship to each other.

#### Dataset

In the case of a Dataset, all map-based variables are filtered using the same bounding box.

```{code-cell} ipython3
ds.em.sub_bbox([-92, 27, -90, 29], drop=True).cf[salt].cf.isel(T=0).cf.sel(Z=0, method='nearest')
```

#### DataArray

```{code-cell} ipython3
ds.cf[salt].em.sub_bbox([-92, 27, -90, 29], drop=True).cf.isel(T=0, Z=-1).cf.plot(x='longitude', y='latitude')
```

### grid point (interpolation and selecting nearest)

Interpolate to a single existing horizontal grid point (and any additional depth and time values for that location) and compare it with method selecting the nearest point to demonstrate we get the same value.

```{code-cell} ipython3
%%time

varname = salt

# Set up a single lon/lat location
j, i = 50, 10
longitude = float(ds.cf[varname].cf['longitude'][j,i])
latitude = float(ds.cf[varname].cf['latitude'][j,i])

# Interpolation
da_out = ds.cf[varname].em.interp2d(longitude, latitude)

# Selection of nearest location in 2D
da_check = ds.cf[varname].em.sel2dcf(longitude=longitude, latitude=latitude).squeeze()

assert np.allclose(da_out, da_check)
```

You could also select a time and/or depth index or interpolate in time and/or depth at the same time:

```{code-cell} ipython3
# Select time index and depth index
ds.cf[varname].em.interp2d(longitude, latitude, iT=0, iZ=0)
```

```{code-cell} ipython3
ds.cf[varname].cf
```

```{code-cell} ipython3
# Interpolate to time value and depth value
ds.cf[varname].em.interp2d(longitude, latitude, T=ds.cf['T'][0], Z=-10)
```

The interpolation is faster the second time the regridder is used — it is saved by the `extract_model` accessor and reused if the lon/lat locations to be interpolated to are the same. Here we interpolate to salinity and it is faster than it was the first time it was used for interpolation the sea surface elevation.

```{code-cell} ipython3
%%time

varname = zeta

# Set up a single lon/lat location
j, i = 50, 10
longitude = float(ds.cf[varname].cf['longitude'][j,i])
latitude = float(ds.cf[varname].cf['latitude'][j,i])

# Interpolation
da_out = ds.cf[varname].em.interp2d(longitude, latitude)

# Selection of nearest location in 2D
da_check = ds.cf[varname].em.sel2dcf(longitude=longitude, latitude=latitude).squeeze()

assert np.allclose(da_out, da_check)
```

### not grid point

+++

#### inside domain  (interpolation and selecting nearest)

For a selected location that is not a grid point (so we can't check it exactly), we show here both interpolating to that location horizontally and selecting the nearest point to that location.

The square in the right hand side plot shows the nearest point selected using `.em.sel2d()` and the circle shows the interpolated value at the exact selected location using `.em.interp2d()`.

```{code-cell} ipython3
varname = zeta

# sel
longitude = -91.49
latitude = 28.510

# isel
iZ = None
iT = 0
isel = dict(T=iT)

# Interpolation
da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ)

# Selection of nearest location in 2D
da_sel = ds.cf[varname].em.sel2dcf(longitude=longitude, latitude=latitude, distances_name="distance").cf.isel(T=iT).squeeze()

# Plot
cmap = ds.cf[varname].cmo.seq
dacheck = ds.cf[varname].cf.isel(isel)
fig, axes = plt.subplots(1, 2, figsize=(15,5))

dacheck.cmo.cfplot(ax=axes[0], x='longitude', y='latitude')
axes[0].scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck.min().values, vmax=dacheck.max().values, cmap=cmap, edgecolors='k')

# make smaller area of model to show
# want model output only within the box defined by these lat/lon values
dacheck_min = dacheck.em.sub_bbox([-91.52, 28.49, -91.49, 28.525], drop=True)
dacheck_min.cmo.cfplot(ax=axes[1], x='longitude', y='latitude')
# interpolation
axes[1].scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck_min.min().values, vmax=dacheck_min.max().values, 
                cmap=cmap, edgecolors='k')
# selection
axes[1].scatter(da_sel.cf['longitude'], da_sel.cf['latitude'], s=50, c=da_sel.cf[varname], 
           vmin=dacheck_min.min().values, vmax=dacheck_min.max().values, 
                cmap=cmap, edgecolors='k', marker='s')
```

We input the extra keyword argument `distances_name` into the call `ds.cf[varname].em.sel2dcf` in order to also return the distance between the requested location and the returned model location. This value is shown here in km:

```{code-cell} ipython3
da_sel["distance"]
```

#### outside domain

+++

Don't extrapolate

This is commented out since it purposefully raises an error:
> ValueError: Longitude outside of available domain. Use extrap=True to extrapolate.

```{code-cell} ipython3
# varname = zeta

# # sel
# longitude = -166
# latitude = 48
# sel = dict(longitude=longitude, latitude=latitude)

# # isel
# iZ = 0
# iT = 0
# isel = dict(Z=iZ, T=iT)

# da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ, extrap=False)

# da_out
```

Extrapolate

```{code-cell} ipython3
varname = zeta

# sel
longitude = -89
latitude = 28.3
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ, extrap=True)

# plot
cmap = ds.cf[varname].cmo.seq
dacheck = ds.cf[varname].cf.isel(isel)
fig, ax = plt.subplots(1,1)
dacheck.cmo.cfplot(ax=ax, x='longitude', y='latitude')
ax.scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck.min().values, vmax=dacheck.max().values, cmap=cmap, edgecolors='k')
```

### points (locstream, interpolation)

Interpolate to unstructured pairs of lon/lat locations instead of grids of lon/lat locations, using `locstream`. Choose grid points so that we can check the accuracy of the results.

```{code-cell} ipython3
:tags: []

varname = zeta

# sel
# this creates 12 pairs of lon/lat points that 
# align with grid points so we can check the 
# interpolation
longitude = ds.cf[varname].cf['longitude'].isel(eta_rho=60, xi_rho=slice(None,None,10))
latitude = ds.cf[varname].cf['latitude'].isel(eta_rho=60, xi_rho=slice(None,None,10))
sel = dict(X=longitude.xi_rho, Y=longitude.eta_rho)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=True)

# check
da_check = ds.cf[varname].cf.isel(isel).cf.sel(sel)

assert np.allclose(da_out, da_check, equal_nan=True)
```

It is not currently possible to interpolate in depth with both more than one time and location. 

This cell is commented out because it purposefully returns an error:
> NotImplementedError: Currently it is not possible to interpolate in depth with more than 1 other (time) dimension.

```{code-cell} ipython3
# ds.cf[salt].em.interp2d(longitude, latitude, Z=-10, locstream=True)
```

### grid of known locations (interpolation)

```{code-cell} ipython3
varname = zeta

# sel
longitude = ds.cf[varname].cf['longitude'][:-50:20,:-200:100]
latitude = ds.cf[varname].cf['latitude'][:-50:20,:-200:100]
sel = dict(X=longitude.xi_rho, Y=longitude.eta_rho)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=False)

# check
da_check = ds.cf[varname].cf.sel(sel).cf.isel(isel)

assert np.allclose(da_out, da_check)
```

### grid of new locations (interpolation, regridding)

```{code-cell} ipython3
varname = zeta

# sel
longitude = np.linspace(ds.cf[varname].cf['longitude'].min(), ds.cf[varname].cf['longitude'].max(), 30)
latitude = np.linspace(ds.cf[varname].cf['latitude'].min(), ds.cf[varname].cf['latitude'].max(), 30)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.cf[varname].em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=False, extrap=False, extrap_val=np.nan)

# plot
cmap = cmo.delta
dacheck = ds.cf[varname].cf.isel(isel)

fig, axes = plt.subplots(1,2, figsize=(10,4))
dacheck.cmo.cfplot(ax=axes[0], x='longitude', y='latitude')
da_out.cmo.cfplot(ax=axes[1], x='longitude', y='latitude')
```

## HYCOM

```{code-cell} ipython3
# url = ['http://tds.hycom.org/thredds/dodsC/GLBy0.08/latest']
# ds = xr.open_mfdataset(url, preprocess=em.preprocess, drop_variables='tau')
# ds.isel(time=slice(0,2)).sel(lat=slice(-20, 30), lon=slice(140,190)).to_netcdf('hycom.nc')
# ds = xr.open_mfdataset('hycom.nc', preprocess=em.preprocess)

url = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'
ds = xr.open_dataset(url, drop_variables='tau')["water_u"].isel(time=slice(0,2), depth=0).sel(lat=slice(-20, 30), lon=slice(140,190))
ds = em.preprocess(ds)
ds = ds.load()
ds
```

```{code-cell} ipython3
ds.cf
```

### grid point

```{code-cell} ipython3
:tags: []


# sel
longitude = float(ds.cf['X'][100])
latitude = float(ds.cf['Y'][150])
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = None
# isel = dict(Z=iZ)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ)

# check
da_check = ds.cf.sel(sel)#.cf.isel(isel)

assert np.allclose(da_out, da_check)
```

### not grid point

+++

#### inside domain

```{code-cell} ipython3

# sel
longitude = 155
latitude = 5
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ)

# plot
cmap = cmo.delta
dacheck = ds.cf.isel(isel)
fig, ax = plt.subplots(1,1)
dacheck.cmo.plot(ax=ax)
ax.scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck.min().values, vmax=dacheck.max().values, cmap=cmap, edgecolors='k')
```

#### outside domain

+++

Don't extrapolate

This purposefully raises an error so is commented out:
> ValueError: Longitude outside of available domain. Use extrap=True to extrapolate.

```{code-cell} ipython3
# # sel
# longitude = -166
# latitude = 48
# sel = dict(longitude=longitude, latitude=latitude)

# # isel
# iZ = None
# iT = 0
# isel = dict(T=iT)

# da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, extrap=False)

# da_out = em.select(**kwargs)
# da_out
```

Extrapolate

```{code-cell} ipython3

# sel
longitude = 139
latitude = 0
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, extrap=True)

# plot
cmap = cmo.delta
dacheck = ds.cf.isel(isel)
fig, ax = plt.subplots(1,1)
dacheck.cmo.plot(ax=ax)
ax.scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck.min().values, vmax=dacheck.max().values, cmap=cmap, edgecolors='k')

ax.set_xlim(138,190)
```

### points (locstream)

Unstructured pairs of lon/lat locations instead of grids of lon/lat locations, using `locstream`.

```{code-cell} ipython3
:tags: []


# sel
# this creates 12 pairs of lon/lat points that 
# align with grid points so we can check the 
# interpolation
longitude = ds.cf['X'][::40].values
latitude = ds.cf['Y'][::80].values
# selecting individual lon/lat locations with advanced xarray indexing
sel = dict(longitude=xr.DataArray(longitude, dims="pts"), latitude=xr.DataArray(latitude, dims="pts"))

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=True)

# check
da_check = ds.cf.isel(isel).cf.sel(sel)

assert np.allclose(da_out, da_check, equal_nan=True)
```

### grid of known locations

```{code-cell} ipython3

# sel
longitude = ds.cf['X'][100::500]
latitude = ds.cf['Y'][100::500]
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = None
# isel = dict(Z=iZ)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=False)

# check
da_check = ds.cf.sel(sel)#.cf.isel(isel)

assert np.allclose(da_out, da_check)
```

### grid of new locations

```{code-cell} ipython3

# sel
longitude = np.linspace(ds.cf['X'].min(), ds.cf['X'].max(), 30)
latitude = np.linspace(ds.cf['Y'].min(), ds.cf['Y'].max(), 30)
sel = dict(longitude=longitude, latitude=latitude)

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=False)
# kwargs = dict(da, longitude=longitude, latitude=latitude, iT=T, iZ=Z)

# da_out = em.select(**kwargs)

# plot
cmap = cmo.delta
dacheck = ds.cf.isel(isel)

fig, axes = plt.subplots(1,2, figsize=(10,4))
dacheck.cmo.plot(ax=axes[0])
da_out.cmo.plot(ax=axes[1])
```

## POM

```{code-cell} ipython3
try:
    url = "https://www.ncei.noaa.gov/thredds/dodsC/model-loofs-agg/Aggregated_LOOFS_Fields_Forecast_best.ncd"
    # url = ['https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/LOOFS/fmrc/Aggregated_7_day_LOOFS_Fields_Forecast_best.ncd']
    # ds = xr.open_mfdataset(url, preprocess=em.preprocess, chunks=None)
    ds= xr.open_dataset(url)
    ds = em.utils.preprocess_pom(ds, interp_vertical=False)
except OSError:
    import pandas as pd
    today = pd.Timestamp.today()
    url = [today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/%Y/%m/%d/glofs.loofs.fields.nowcast.%Y%m%d.t00z.nc'),
           today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/%Y/%m/%d/glofs.loofs.fields.nowcast.%Y%m%d.t06z.nc')]
    ds = xr.open_mfdataset(url, preprocess=em.preprocess)

ds = ds["zeta"].isel(time=slice(0,2)).load()
ds
```

```{code-cell} ipython3
ds.cf
```

### grid point

```{code-cell} ipython3
%%time

# Set up a single lon/lat location
j, i = 10, 10
longitude = float(ds.cf['longitude'][j,i])
latitude = float(ds.cf['latitude'][j,i])

# Select-by-index a time index and no vertical index (zeta has none)
# also lon/lat by index
Z = None
iT = 0
isel = dict(T=iT, X=i, Y=j)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=Z)

# check work
da_check = ds.cf.isel(isel)

assert np.allclose(da_out, da_check)
```

This is faster the second time the regridder is used — it is saved by the `extract_model` accessor and reused if the lon/lat locations to be interpolated to are the same.

+++

### not grid point

+++

#### inside domain

```{code-cell} ipython3

# sel
longitude = -78.0
latitude = 43.6

# isel
iZ = None
iT = 1
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ)

# plot
cmap = ds.cmo.seq
dacheck = ds.cf.isel(isel)
fig, ax = plt.subplots(1,1)
dacheck.cmo.cfplot(ax=ax, x='longitude', y='latitude')
ax.scatter(da_out.cf['longitude'], da_out.cf['latitude'], s=50, c=da_out, 
           vmin=dacheck.min().values, vmax=dacheck.max().values, cmap=cmap, edgecolors='k')
```

### points (locstream)

Unstructured pairs of lon/lat locations instead of grids of lon/lat locations, using `locstream`.

```{code-cell} ipython3
:tags: []


# sel
# this creates 12 pairs of lon/lat points that 
# align with grid points so we can check the 
# interpolation
longitude = ds.cf['longitude'].cf.isel(Y=20, X=slice(None, None, 10))
latitude = ds.cf['latitude'].cf.isel(Y=20, X=slice(None, None, 10))
sel = dict(X=longitude.cf['X'], Y=longitude.cf['Y'])

# isel
iZ = None
iT = 0
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=True)

# check
da_check = ds.cf.isel(isel).cf.sel(sel)

assert np.allclose(da_out, da_check, equal_nan=True)
```

### grid of new locations

```{code-cell} ipython3

# sel
longitude = np.linspace(ds.cf['longitude'].min(), ds.cf['longitude'].max(), 15)
latitude = np.linspace(ds.cf['latitude'].min(), ds.cf['latitude'].max(), 15)

# isel
iZ = None
iT = 1
isel = dict(T=iT)

da_out = ds.em.interp2d(longitude, latitude, iT=iT, iZ=iZ, locstream=False, extrap=False, extrap_val=np.nan)

# plot
cmap = cmo.delta
dacheck = ds.cf.isel(isel)

fig, axes = plt.subplots(1,2, figsize=(10,4))
dacheck.cmo.cfplot(ax=axes[0], x='longitude', y='latitude')
da_out.cmo.cfplot(ax=axes[1], x='longitude', y='latitude')
```

```{code-cell} ipython3

```
