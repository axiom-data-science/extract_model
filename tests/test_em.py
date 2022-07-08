from pathlib import Path
from time import time

import numpy as np
import pytest
import xarray as xr

import extract_model as em

from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


def test_T_interp_no_xesmf():
    """Test interpolation in time for one model.

    Also test for no xESMF package."""

    url = Path(__file__).parent / "test_roms.nc"
    ds = xr.open_dataset(url)
    da_out, _ = em.select(da=ds["zeta"], T=0.5)
    assert np.allclose(da_out[0, 0], -0.12584045)

    XESMF_AVAILABLE = em.extract_model.XESMF_AVAILABLE
    em.extract_model.XESMF_AVAILABLE = False
    da_out, _ = em.select(da=ds["zeta"], T=0.5)
    assert np.allclose(da_out[0, 0], -0.12584045)
    em.extract_model.XESMF_AVAILABLE = XESMF_AVAILABLE


def test_Z_interp():
    """Test interpolation in depth for one model."""

    url = Path(__file__).parent / "test_hycom.nc"
    ds = xr.open_dataset(url)
    da_out, _ = em.select(da=ds["water_u"], Z=1.0)
    assert np.allclose(da_out[-1, -1], -0.1365)


def test_hor_interp_no_xesmf():
    """Code shouldn't work without xESMF.

    Make sure horizontal interpolation doesn't run if xESMF not available."""

    da = models[3]["da"]
    i, j = models[3]["i"], models[3]["j"]

    if da.cf["longitude"].ndim == 1:
        longitude = float(da.cf["X"][i])
        latitude = float(da.cf["Y"][j])

    elif da.cf["longitude"].ndim == 2:
        longitude = float(da.cf["longitude"][j, i])
        latitude = float(da.cf["latitude"][j, i])

    XESMF_AVAILABLE = em.extract_model.XESMF_AVAILABLE
    em.extract_model.XESMF_AVAILABLE = False
    with pytest.raises(ModuleNotFoundError):
        em.select(da, longitude=longitude, latitude=latitude, T=0.5)
    em.extract_model.XESMF_AVAILABLE = XESMF_AVAILABLE


@pytest.mark.parametrize("model", models)
class TestModel:
    def test_grid_point_isel_Z(self, model):
        """Select and return a grid point.

        Also make sure weights are used if input."""

        da = model["da"]
        i, j = model["i"], model["j"]
        Z, T = model["Z"], model["T"]

        if da.cf["longitude"].ndim == 1:
            longitude = float(da.cf["X"][i])
            latitude = float(da.cf["Y"][j])

        elif da.cf["longitude"].ndim == 2:
            longitude = float(da.cf["longitude"][j, i])
            latitude = float(da.cf["latitude"][j, i])

        da_check = da.em.sel2d(longitude, latitude, iT=T, iZ=Z)

        # save time required when regridder is being calculated
        try:
            ta0 = time()
            da_out = da.em.interp2d(lons=longitude, lats=latitude, iZ=Z, iT=T)
            ta1 = time() - ta0

            assert np.allclose(da_out, da_check)

            # Make sure weights are reused when present
            # here they are used from being saved in the da object
            tb0 = time()
            da_out = da.em.interp2d(lons=longitude, lats=latitude, iZ=Z, iT=T)
            tb1 = time() - tb0

            # speed up should be at least 1.5 times
            assert ta1 / tb1 > 1.5

            # here they are used explicitly
            weights = list(da.em.weights_map.values())[0]
            da2 = model["da"].copy()
            tc0 = time()
            da_out = da2.em.interp2d(
                lons=longitude, lats=latitude, iZ=Z, iT=T, weights=weights
            )
            tc1 = time() - tc0

            # speed up should be at least 1.5 times
            assert ta1 / tc1 > 1.5

        # this should only run if xESMF is installed
        except AttributeError:
            if not em.extract_model.XESMF_AVAILABLE:
                pass

    def test_extrap_False(self, model):
        """Search for point outside domain, which should raise an assertion."""

        da = model["da"]
        lon1, lat1 = model["lon1"], model["lat1"]
        Z, T = model["Z"], model["T"]

        # sel
        longitude = lon1
        latitude = lat1

        kwargs = dict(
            lons=longitude,
            lats=latitude,
            iT=T,
            iZ=Z,
            extrap=False,
        )

        with pytest.raises(ValueError):
            da.em.interp2d(**kwargs)

    def test_extrap_True(self, model):
        """Check that a point right outside domain has
        extrapolated value of neighbor point."""

        da = model["da"]
        i, j = model["i"], model["j"]
        Z, T = model["Z"], model["T"]

        if da.cf["longitude"].ndim == 1:
            longitude_check = float(da.cf["X"][i])
            longitude = longitude_check - 0.1
            latitude = float(da.cf["Y"][j])

        elif da.cf["longitude"].ndim == 2:
            longitude = float(da.cf["longitude"][j, i])
            latitude = float(da.cf["latitude"][j, i])

        kwargs = dict(
            lons=longitude,
            lats=latitude,
            iZ=Z,
            iT=T,
            extrap=True,
        )

        try:
            da_out = da.em.interp2d(**kwargs)
            da_check = da.em.sel2d(longitude, latitude, iT=T, iZ=Z)

            assert np.allclose(da_out, da_check, equal_nan=True)

        # this should only run if xESMF is installed
        except AttributeError:
            if not em.extract_model.XESMF_AVAILABLE:
                pass

    def test_extrap_False_extrap_val_nan(self, model):
        """Check that land point returns np.nan for extrap=False
        and extrap_val=np.nan."""

        da = model["da"]
        lon2, lat2 = model["lon2"], model["lat2"]
        Z, T = model["Z"], model["T"]

        # sel
        longitude = lon2
        latitude = lat2

        kwargs = dict(
            lons=longitude,
            lats=latitude,
            iZ=Z,
            iT=T,
            extrap=False,
            extrap_val=np.nan,
        )

        try:
            da_out = da.em.interp2d(**kwargs)
            assert da_out.isnull()
        # this should only run if xESMF is installed
        except AttributeError:
            if not em.extract_model.XESMF_AVAILABLE:
                pass

    def test_locstream(self, model):

        da = model["da"]
        lonslice, latslice = model["lonslice"], model["latslice"]
        Z, T = model["Z"], model["T"]

        if da.cf["longitude"].ndim == 1:
            longitude = da.cf["X"][lonslice].values
            latitude = da.cf["Y"][latslice].values
            sel = dict(
                longitude=xr.DataArray(longitude, dims="pts"),
                latitude=xr.DataArray(latitude, dims="pts"),
            )
            isel = dict(Z=Z)

        elif da.cf["longitude"].ndim == 2:
            longitude = da.cf["longitude"].cf.isel(Y=50, X=lonslice)
            latitude = da.cf["latitude"].cf.isel(Y=50, X=lonslice)
            isel = dict(T=T)
            sel = dict(X=longitude.cf["X"], Y=longitude.cf["Y"])

        kwargs = dict(
            lons=longitude,
            lats=latitude,
            iZ=Z,
            iT=T,
            locstream=True,
        )

        try:
            da_out = da.em.interp2d(**kwargs)
            da_check = da.cf.sel(sel).cf.isel(isel)
            assert np.allclose(da_out, da_check, equal_nan=True)
        # this should only run if xESMF is installed
        except AttributeError:
            if not em.extract_model.XESMF_AVAILABLE:
                pass

    def test_grid(self, model):

        da = model["da"]
        lonslice, latslice = model["lonslice"], model["latslice"]
        Z, T = model["Z"], model["T"]

        if da.cf["longitude"].ndim == 1:
            longitude = da.cf["X"][lonslice]
            latitude = da.cf["Y"][latslice]
            sel = dict(longitude=longitude, latitude=latitude)
            isel = dict(Z=Z)
            da_check = da.cf.sel(sel).cf.isel(isel)

        elif da.cf["longitude"].ndim == 2:
            longitude = da.cf["longitude"][latslice, lonslice].values
            latitude = da.cf["latitude"][latslice, lonslice].values
            isel = dict(T=T, X=lonslice, Y=latslice)
            da_check = da.cf.isel(isel)

        kwargs = dict(lons=longitude, lats=latitude, iZ=Z, iT=T, locstream=False)

        try:
            da_out = da.em.interp2d(**kwargs)
            assert np.allclose(da_out, da_check)
        # this should only run if xESMF is installed
        except AttributeError:
            if not em.extract_model.XESMF_AVAILABLE:
                pass

    def test_preprocess(self, model):
        """Test preprocessing on output."""

        da = model["da"]
        axes = ["T", "Z", "Y", "X"]
        conds = [True if axis in da.cf.axes else axis for axis in axes]

        assert all(conds)
