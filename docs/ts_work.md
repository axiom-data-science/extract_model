---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3.9.13 ('extract_model')
  language: python
  name: python3
---

# Time Series Extraction

```{code-cell} ipython3
import xarray as xr
import cf_xarray
import extract_model as em
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import cmocean.cm as cmo

# For this notebook, it's nicer if we don't show the array values by default
xr.set_options(display_expand_data=False)
xr.set_options(display_expand_coords=False)
xr.set_options(display_expand_attrs=False)
```

## Example model to use

```{code-cell} ipython3
:tags: []

# !wget https://www.ncei.noaa.gov/thredds/fileServer/model-ciofs-files/2022/03/nos.ciofs.fields.n001.20220301.t12z.nc
# !wget https://www.ncei.noaa.gov/thredds/fileServer/model-ciofs-files/2022/03/nos.ciofs.fields.n001.20220301.t18z.nc
```

```{code-cell} ipython3
:tags: []

# Structured: CIOFS: ROMS Cook Inlet model
# get some model output locally
loc1 = glob('nos.ciofs.*.nc')


# # Unstructured: CREOFS: SELFE Columbia River model
# today = pd.Timestamp.today()
# loc2 = [today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CREOFS/MODELS/%Y/%m/%d/nos.creofs.fields.n000.%Y%m%d.t03z.nc'),
#         today.strftime('https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/CREOFS/MODELS/%Y/%m/%d/nos.creofs.fields.n001.%Y%m%d.t03z.nc')]
```

```{code-cell} ipython3
:tags: []

ds1 = xr.open_mfdataset(loc1, preprocess=em.preprocess)
ds1
```

## Demo code

+++

### Select time series from nearest point

+++

Use a DataArray or a Dataset, but keep in mind that when there are multiple horizontal grids (like there are for ROMS models), you will need to specify which grid's longitude and latitude coordinates to use. The API is meant to be analogous to that of selecting with `xarray` using `.sel()`.

This functionality uses [`xoak`](https://xoak.readthedocs.io/en/latest/).

```{code-cell} ipython3
da1 = ds1['temp']
lon0, lat0 = -151.4, 59  # cook inlet
```

For any of the following results, access the depth values with

```
[output].cf['vertical'].values
```

+++

#### 2D lon/lat

+++

The first request will take longer than a second request would because the second request uses the index calculated the first time.

```{code-cell} ipython3
%%time
output = da1.em.sel2d(lon_rho=lon0, lat_rho=lat0).squeeze()
output
```

```{code-cell} ipython3
%%time
output = da1.em.sel2d(lon_rho=lon0, lat_rho=lat0).squeeze()
output
```

Access the associated indices:

```{code-cell} ipython3
j, i = int(output.eta_rho.values), int(output.xi_rho.values)
```

Profile for first time matches:

```{code-cell} ipython3
output.cf.isel(T=0).cf.plot(y='vertical', lw=4)
da1.cf.isel(X=i, Y=j, T=0).cf.plot(y='vertical', lw=2)
```

Surface value for first time matches map:

```{code-cell} ipython3
mappable = da1.cf.isel(T=0, Z=-1).cf.plot(x='longitude', y='latitude')
vmin, vmax = mappable.get_clim()
plt.scatter(lon0, lat0, c=output.cf.isel(T=0, Z=-1).values, cmap=mappable.cmap, vmin=vmin, vmax=vmax, edgecolors='k')
```

To retrieve the values:

`output.values`

```{code-cell} ipython3
output.values
```

To retrieve the associated depths:

`output.cf['vertical'].values`

```{code-cell} ipython3
output.cf['vertical'].values
```

#### 3D lon/lat/Z or iZ

+++

Return model output nearest to lon, lat, Z value. `z_rho` has two values because the depth changes in time.

```{code-cell} ipython3
out = da1.em.sel2d(lon_rho=lon0, lat_rho=lat0).squeeze()
out
```

```{code-cell} ipython3
out.em.selZ(depths=-40)
```

Return model output nearest to lon, lat, at index iZ in Z dimension.

```{code-cell} ipython3
da1.em.sel2d(lon_rho=lon0, lat_rho=lat0).cf.isel(Z=-1)
```

### Interpolate time series at exact point

```{code-cell} ipython3
da1 = ds1['salt']
lon0, lat0 = -152, 58
lons, lats = [-151, -152], [59,58]
```

#### 2D lon/lat

+++

1 lon/lat pair

```{code-cell} ipython3
%%time
output = da1.em.interp2d(lon0, lat0)
output
```

Surface value for first time matches map:

```{code-cell} ipython3
cmap=cmo.haline
mappable = da1.cf.isel(T=0, Z=-1).cf.plot(x='longitude', y='latitude', cmap=cmap)
vmin, vmax = mappable.get_clim()
plt.scatter(lon0, lat0, c=output.cf.isel(T=0, Z=-1).values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='k')
```

To retrieve the values:

`output.values`

```{code-cell} ipython3
output.values
```

To retrieve the associated depths:

`output.cf['vertical'].values`

```{code-cell} ipython3
output.cf['vertical'].values
```

multiple lon/lat pairs

```{code-cell} ipython3
%%time
da1.em.interp2d(lons, lats)
```

#### 3D: lon, lat, iZ

+++

Return model output interpolated to lon, lat, Z value.

```{code-cell} ipython3
da1.em.interp2d(lon0, lat0, Z=-40)
```

Return model output interpolated to lon, lat, at index iZ in Z dimension.

```{code-cell} ipython3
da1.em.interp2d(lon0, lat0, iZ=-1)
```

Note that it is not currently possible to interpolate in depth when there are both multiple times and locations.

If uncommented, the following cell will return:
> NotImplementedError: Currently it is not possible to interpolate in depth with more than 1 other (time) dimension.

```{code-cell} ipython3
# da1.em.interp2d(lons, lats, Z=-40)
```
