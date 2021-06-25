from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import extract_model as em


models = []

# MOM6 inputs
url = Path(__file__).parent / "test_mom6.nc"
ds = xr.open_dataset(url)
ds = ds.cf.guess_coord_axis()
varname = "u"
i, j = 0, 0
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = -149.0, 56.0
lonslice = slice(None, 5)
latslice = slice(None, 5)
model_names = [None, "sea_water_x_velocity", None, None, None]
mom6 = dict(
    ds=ds,
    varname=varname,
    i=i,
    j=j,
    Z=Z,
    T=T,
    lon1=lon1,
    lat1=lat1,
    lon2=lon2,
    lat2=lat2,
    lonslice=lonslice,
    latslice=latslice,
    model_names=model_names,
)
models += [mom6]

# HYCOM inputs
url = Path(__file__).parent / "test_hycom.nc"
ds = xr.open_dataset(url)
varname = "u"
i, j = 0, 30
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = 149.0, -10.1
lonslice = slice(10, 15)
latslice = slice(10, 15)
model_names = [None, "eastward_sea_water_velocity", None, None, None]
hycom = dict(
    ds=ds,
    varname=varname,
    i=i,
    j=j,
    Z=Z,
    T=T,
    lon1=lon1,
    lat1=lat1,
    lon2=lon2,
    lat2=lat2,
    lonslice=lonslice,
    latslice=latslice,
    model_names=model_names,
)
models += [hycom]

# Second HYCOM example inputs, from Heather
url = Path(__file__).parent / "test_hycom2.nc"
ds = xr.open_dataset(url)
varname = "u"
j, i = 30, 0
Z, T = 0, None
lon1, lat1 = -166, 48
lon2, lat2 = -91, 29.5
lonslice = slice(10, 15)
latslice = slice(10, 15)
model_names = [None, "eastward_sea_water_velocity", None, None, None]
hycom2 = dict(
    ds=ds,
    varname=varname,
    i=i,
    j=j,
    Z=Z,
    T=T,
    lon1=lon1,
    lat1=lat1,
    lon2=lon2,
    lat2=lat2,
    lonslice=lonslice,
    latslice=latslice,
    model_names=model_names,
)
models += [hycom2]

# ROMS inputs
url = Path(__file__).parent / "test_roms.nc"
ds = xr.open_dataset(url)
varname = "ssh"
j, i = 50, 10
Z1, T = None, 0
lon1, lat1 = -166, 48
lon2, lat2 = -91, 29.5
lonslice = slice(10, 15)
latslice = slice(10, 15)
model_names = ["sea_surface_elevation", None, None, None, None]
roms = dict(
    ds=ds,
    varname=varname,
    i=i,
    j=j,
    Z=Z1,
    T=T,
    lon1=lon1,
    lat1=lat1,
    lon2=lon2,
    lat2=lat2,
    lonslice=lonslice,
    latslice=latslice,
    model_names=model_names,
)
models += [roms]


def test_T_interp():
    """Test interpolation in time for one model."""

    url = Path(__file__).parent / "test_roms.nc"
    ds = xr.open_dataset(url)
    dr = em.select(ds=ds, T=0.5)
    assert np.allclose(dr["zeta"][0, 0], -0.12584045)


def test_Z_interp():
    """Test interpolation in depth for one model."""

    url = Path(__file__).parent / "test_hycom.nc"
    ds = xr.open_dataset(url)
    dr = em.select(ds=ds, Z=1.0)
    assert np.allclose(dr["water_u"][-1, -1], -0.1365)


@pytest.mark.parametrize("model", models)
class TestModel:
    def test_grid_point_isel_Z(self, model):
        """Select and return a grid point."""

        ds = model["ds"]
        varname = model["varname"]
        i, j = model["i"], model["j"]
        Z, T = model["Z"], model["T"]

        if ds.cf["longitude"].ndim == 1:
            longitude = float(ds.cf[varname].cf["X"][i])
            latitude = float(ds.cf[varname].cf["Y"][j])
            sel = dict(longitude=longitude, latitude=latitude)

            # isel
            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[varname].cf.sel(sel).cf.isel(isel)
        elif ds.cf["longitude"].ndim == 2:
            longitude = float(ds.cf[varname].cf["longitude"][j, i])
            latitude = float(ds.cf[varname].cf["latitude"][j, i])

            isel = dict(T=T, X=i, Y=j)

            # check
            dr_check = ds.cf[varname].cf.isel(isel)

        kwargs = dict(
            ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname
        )

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check)

    def test_extrap_False(self, model):
        """Search for point outside domain, which should raise an assertion."""

        ds = model["ds"]
        varname = model["varname"]
        lon1, lat1 = model["lon1"], model["lat1"]
        Z, T = model["Z"], model["T"]

        # sel
        longitude = lon1
        latitude = lat1
        sel = dict(longitude=longitude, latitude=latitude)

        # isel
        isel = dict(Z=Z, T=T)

        kwargs = dict(
            ds=ds,
            longitude=longitude,
            latitude=latitude,
            iT=T,
            iZ=Z,
            varname=varname,
            extrap=False,
        )

        with pytest.raises(AssertionError):
            em.select(**kwargs)

    def test_extrap_True(self, model):
        """Check that a point right outside domain has
        extrapolated value of neighbor point."""

        ds = model["ds"]
        varname = model["varname"]
        i, j = model["i"], model["j"]
        Z, T = model["Z"], model["T"]

        if ds.cf["longitude"].ndim == 1:
            longitude_check = float(ds.cf[varname].cf["X"][i])
            longitude = longitude_check - 0.1
            latitude = float(ds.cf[varname].cf["Y"][j])
            sel = dict(longitude=longitude_check, latitude=latitude)

            # isel
            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[varname].cf.sel(sel).cf.isel(isel)
        elif ds.cf["longitude"].ndim == 2:
            longitude = float(ds.cf[varname].cf["longitude"][j, i])
            latitude = float(ds.cf[varname].cf["latitude"][j, i])

            isel = dict(T=T, X=i, Y=j)

            # check
            dr_check = ds.cf[varname].cf.isel(isel)

        kwargs = dict(
            ds=ds,
            longitude=longitude,
            latitude=latitude,
            iZ=Z,
            iT=T,
            varname=varname,
            extrap=True,
        )

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check, equal_nan=True)

    def test_extrap_False_extrap_val_nan(self, model):
        """Check that land point returns np.nan for extrap=False
        and extrap_val=np.nan."""

        ds = model["ds"]
        varname = model["varname"]
        lon2, lat2 = model["lon2"], model["lat2"]
        Z, T = model["Z"], model["T"]

        # sel
        longitude = lon2
        latitude = lat2

        # isel
        isel = dict(Z=Z, T=T)

        kwargs = dict(
            ds=ds,
            longitude=longitude,
            latitude=latitude,
            iZ=Z,
            iT=T,
            varname=varname,
            extrap=False,
            extrap_val=np.nan,
        )

        dr = em.select(**kwargs)

        assert dr.isnull()

    def test_locstream(self, model):

        ds = model["ds"]
        varname = model["varname"]
        lonslice, latslice = model["lonslice"], model["latslice"]
        Z, T = model["Z"], model["T"]

        if ds.cf["longitude"].ndim == 1:
            longitude = ds.cf[varname].cf["X"][lonslice].values
            latitude = ds.cf[varname].cf["Y"][latslice].values
            sel = dict(
                longitude=xr.DataArray(longitude, dims="pts"),
                latitude=xr.DataArray(latitude, dims="pts"),
            )
            isel = dict(Z=Z)

        elif ds.cf["longitude"].ndim == 2:
            longitude = ds.cf[varname].cf["longitude"].cf.isel(Y=50, X=lonslice)
            latitude = ds.cf[varname].cf["latitude"].cf.isel(Y=50, X=lonslice)
            isel = dict(T=T)
            sel = dict(X=longitude.cf["X"], Y=longitude.cf["Y"])

        kwargs = dict(
            ds=ds,
            longitude=longitude,
            latitude=latitude,
            iZ=Z,
            iT=T,
            varname=varname,
            locstream=True,
        )

        dr = em.select(**kwargs)

        # check
        dr_check = ds.cf[varname].cf.sel(sel).cf.isel(isel)

        assert np.allclose(dr, dr_check, equal_nan=True)

    def test_grid(self, model):

        ds = model["ds"]
        varname = model["varname"]
        lonslice, latslice = model["lonslice"], model["latslice"]
        Z, T = model["Z"], model["T"]

        if ds.cf["longitude"].ndim == 1:
            longitude = ds.cf[varname].cf["X"][lonslice]
            latitude = ds.cf[varname].cf["Y"][latslice]
            sel = dict(longitude=longitude, latitude=latitude)

            isel = dict(Z=Z)

            # check
            dr_check = ds.cf[varname].cf.sel(sel).cf.isel(isel)

        elif ds.cf["longitude"].ndim == 2:
            longitude = ds.cf[varname].cf["longitude"][latslice, lonslice].values
            latitude = ds.cf[varname].cf["latitude"][latslice, lonslice].values

            isel = dict(T=T, X=lonslice, Y=latslice)

            # check
            dr_check = ds.cf[varname].cf.isel(isel)

        kwargs = dict(
            ds=ds, longitude=longitude, latitude=latitude, iZ=Z, iT=T, varname=varname
        )

        dr = em.select(**kwargs)

        assert np.allclose(dr, dr_check)
