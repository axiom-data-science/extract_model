from extract_model import extract_model as em
import xarray as xr
import numpy as np
import pytest
from pathlib import Path

models = []

# base_path = Path.home() / "tests/"
# Path(em.__file__).parent.parent / 'tests' / 'test_mom6.nc'

# MOM6 inputs
# url = base_path / "test_mom6.nc"
url = Path(__file__).parent / 'test_mom6.nc'
# url = 'tests/test_mom6.nc'
ds = xr.open_dataset(url)
varname = 'u'
cf_var = em.get_var_cf(ds, varname)
i, j = 0, 0
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = -149, 56
lonslice = slice(None,5)
latslice = slice(None,5)
mom6 = dict(ds=ds, varname=varname, cf_var=cf_var,
              i=i, j=j, Z=Z, T=T, lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2,
              lonslice=lonslice, latslice=latslice
)
models += [mom6]

# HYCOM inputs
url = Path(__file__).parent / 'test_hycom.nc'
# url = base_path / "test_hycom.nc"
# url = 'test_hycom.nc'
ds = xr.open_dataset(url)
varname = 'u'
cf_var = em.get_var_cf(ds, varname)
i, j = 0, 30
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = 149, -10.1
lonslice = slice(10,15)
latslice = slice(10,15)
hycom = dict(ds=ds, varname=varname, cf_var=cf_var,
              i=i, j=j, Z=Z, T=T, lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2,
              lonslice=lonslice, latslice=latslice
)
models += [hycom]

# Second HYCOM example inputs, from Heather
url = Path(__file__).parent / 'test_hycom2.nc'
# url = base_path / "test_hycom2.nc"
# url = 'test_hycom2.nc'
ds = xr.open_dataset(url)
varname = 'u'
cf_var = em.get_var_cf(ds, varname)
j, i = 30, 0
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = -91, 29.5
lonslice = slice(10,15)
latslice = slice(10,15)
hycom2 = dict(ds=ds, varname=varname, cf_var=cf_var,
              i=i, j=j, Z=Z, T=T, lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2,
              lonslice=lonslice, latslice=latslice
)
models += [hycom2]

# ROMS inputs
# url = base_path / "test_roms.nc"
url = Path(__file__).parent / 'test_roms.nc'
# url = 'test_roms.nc'
ds = xr.open_dataset(url)
varname = 'ssh'
cf_var = em.get_var_cf(ds, varname)
j, i = 50, 10
Z, T = None, 0
lon1, lat1 = -166, 48
lon2, lat2 = -91, 29.5
lonslice = slice(10,15)
latslice = slice(10,15)
roms = dict(ds=ds, varname=varname, cf_var=cf_var,
              i=i, j=j, Z=Z, T=T, lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2,
              lonslice=lonslice, latslice=latslice
)
models += [roms]

@pytest.mark.parametrize("model", models)
class TestModel:
    def test_grid_point_isel_Z(self, model):
        """Select and return a grid point."""

        ds = model['ds']
        varname = model['varname']
        cf_var = model['cf_var']
        i, j = model['i'], model['j']
        Z, T = model['Z'], model['T']

        if ds.cf['longitude'].ndim == 1:
            longitude = float(ds.cf[cf_var].cf['X'][i])
            latitude = float(ds.cf[cf_var].cf['Y'][j])
            sel = dict(longitude=longitude, latitude=latitude)

            # isel
            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[cf_var].cf.sel(sel).cf.isel(isel)
        elif ds.cf['longitude'].ndim == 2:
            longitude = float(ds.cf[cf_var].cf['longitude'][j,i])
            latitude = float(ds.cf[cf_var].cf['latitude'][j,i])

            isel = dict(T=T, X=i, Y=j)

            # check
            dr_check = ds.cf[cf_var].cf.isel(isel)

        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname)

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check)


#     def test_grid_point_interp_Z(self, model):
#         """Select and return a grid point."""

#         ds = model['ds']
#         varname = model['varname']
#         cf_var = model['cf_var']
#         i, j = model['i'], model['j']
#         Z, T = model['Z'], model['T']

#         if ds.cf['longitude'].ndim == 1:
#             longitude = float(ds.cf[cf_var].cf['X'][i])
#             latitude = float(ds.cf[cf_var].cf['Y'][j])
#             sel = dict(longitude=longitude, latitude=latitude, Z=Z)

#             # isel
#             isel = dict(Z=Z)

#             # check
#             dr_check = ds.cf[cf_var].cf.sel(sel).cf.isel(isel)
#         elif ds.cf['longitude'].ndim == 2:
#             longitude = float(ds.cf[cf_var].cf['longitude'][j,i])
#             latitude = float(ds.cf[cf_var].cf['latitude'][j,i])

#             isel = dict(T=T, X=i, Y=j)

#             # check
#             dr_check = ds.cf[cf_var].cf.isel(isel)

# #         # sel
# #         longitude = float(ds.cf[cf_var].cf['X'][i])
# #         latitude = float(ds.cf[cf_var].cf['Y'][j])
# # #         Z = float(ds.cf[cf_var].cf['Z'][0])
# #         sel = dict(longitude=longitude, latitude=latitude, Z=Z)

#         kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, Z=Z, varname=varname)

#         dr = em.select(**kwargs)

#         # check
# #         dr_check = ds.cf[cf_var].cf.sel(sel)

#         assert np.allclose(dr, dr_check)


    def test_extrap_False(self, model):
        """Search for point outside domain, which should raise an assertion."""

        ds = model['ds']
        varname = model['varname']
        lon1, lat1 = model['lon1'], model['lat1']
        Z, T = model['Z'], model['T']

        # sel
        longitude = lon1
        latitude = lat1
        sel = dict(longitude=longitude, latitude=latitude)

        # isel
        isel = dict(Z=Z, T=T)

        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iT=T, iZ=Z, varname=varname, extrap=False)

        with pytest.raises(AssertionError):
            em.select(**kwargs)


    def test_extrap_True(self, model):
        '''Check that a point right outside domain has
        extrapolated value of neighbor point.'''

        ds = model['ds']
        varname = model['varname']
        cf_var = model['cf_var']
        i, j = model['i'], model['j']
        Z, T = model['Z'], model['T']

        if ds.cf['longitude'].ndim == 1:
            longitude_check = float(ds.cf[cf_var].cf['X'][i])
            longitude = longitude_check - 0.1
            latitude = float(ds.cf[cf_var].cf['Y'][j])
            sel = dict(longitude=longitude_check, latitude=latitude)

            # isel
            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[cf_var].cf.sel(sel).cf.isel(isel)
        elif ds.cf['longitude'].ndim == 2:
            longitude = float(ds.cf[cf_var].cf['longitude'][j,i])
            latitude = float(ds.cf[cf_var].cf['latitude'][j,i])

            isel = dict(T=T, X=i, Y=j)

            # check
            dr_check = ds.cf[cf_var].cf.isel(isel)


        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname, extrap=True)

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check, equal_nan=True)


    def test_extrap_False_extrap_val_nan(self, model):
        """Check that land point returns np.nan for extrap=False
        and extrap_val=np.nan."""

        ds = model['ds']
        varname = model['varname']
        lon2, lat2 = model['lon2'], model['lat2']
        Z, T = model['Z'], model['T']

        # sel
        longitude = lon2
        latitude = lat2

        # isel
        isel = dict(Z=Z, T=T)

        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname, extrap=False, extrap_val=np.nan)

        dr = em.select(**kwargs)

        assert dr.isnull()


    def test_locstream(self, model):

        ds = model['ds']
        varname = model['varname']
        lonslice, latslice = model['lonslice'], model['latslice']
        Z, T = model['Z'], model['T']

        cf_var = em.get_var_cf(ds, varname)

        if ds.cf['longitude'].ndim == 1:
            longitude = ds.cf[cf_var].cf['X'][lonslice].values
            latitude = ds.cf[cf_var].cf['Y'][latslice].values
            sel = dict(longitude=xr.DataArray(longitude, dims="pts"), latitude=xr.DataArray(latitude, dims="pts"))
            isel = dict(Z=Z)

        elif ds.cf['longitude'].ndim == 2:
            longitude = ds.cf[cf_var].cf['longitude'].cf.isel(Y=50, X=lonslice)
            latitude = ds.cf[cf_var].cf['latitude'].cf.isel(Y=50, X=lonslice)
            isel = dict(T=T)
            sel = dict(X=longitude.cf['X'], Y=longitude.cf['Y'])

        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T,
                      varname=varname, locstream=True)

        dr = em.select(**kwargs)

        # check
        dr_check = ds.cf[cf_var].cf.sel(sel).cf.isel(isel)

        assert np.allclose(dr, dr_check, equal_nan=True)


    def test_grid(self, model):

        ds = model['ds']
        varname = model['varname']
        lonslice, latslice = model['lonslice'], model['latslice']
        Z, T = model['Z'], model['T']

        cf_var = em.get_var_cf(ds, varname)

        if ds.cf['longitude'].ndim == 1:
            longitude = ds.cf[cf_var].cf['X'][lonslice]
            latitude = ds.cf[cf_var].cf['Y'][latslice]
            sel = dict(longitude=longitude, latitude=latitude)

            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[cf_var].cf.sel(sel).cf.isel(isel)

        elif ds.cf['longitude'].ndim == 2:
            longitude = ds.cf[cf_var].cf['longitude'][latslice,lonslice].values
            latitude = ds.cf[cf_var].cf['latitude'][latslice,lonslice].values

            isel = dict(T=T, X=lonslice, Y=latslice)

            # check
            dr_check = ds.cf[cf_var].cf.isel(isel)

        kwargs = dict(ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname)

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check)
