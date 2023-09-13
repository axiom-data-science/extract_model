#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""
Started this as a separate file to avoid figuring out test merging
knowing that other test work has been merged.
Need to combine later to be more organized.
"""
import warnings

from pathlib import Path
from time import time

import numpy as np
import pytest

import extract_model as em

from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)

# ROMS example output
model = models[3]
da = model["da"]
i, j = model["i"], model["j"]
lon = float(da.cf["longitude"][j, i])
lat = float(da.cf["latitude"][j, i])


def test_lon_lat_types():
    """Test sel2d with different types for lon/lat"""

    da_check = da.cf.isel(X=i, Y=j)

    # Floats
    da_test = em.sel2d(da, lon_rho=lon, lat_rho=lat).squeeze()
    assert np.allclose(da_check, da_test.to_array())

    # List
    da_test = em.sel2d(da, lon_rho=[lon], lat_rho=[lat]).squeeze()
    assert np.allclose(da_check, da_test.to_array())

    # Array
    da_test = em.sel2d(da, lon_rho=np.array([lon]), lat_rho=np.array([lat])).squeeze()
    assert np.allclose(da_check, da_test.to_array())


def test_2D():
    """Make sure 2D works"""

    # Make 2D grid of indices
    ii, jj = [i, i + 1, i + 2], [j, j + 1, j + 2]
    I, J = np.meshgrid(ii, jj)
    Lon = da.cf["longitude"].cf.isel(X=I.flatten(), Y=J.flatten())
    Lat = da.cf["latitude"].cf.isel(X=I.flatten(), Y=J.flatten())
    Lon, Lat = Lon.values, Lat.values

    da_check = da.cf.isel(X=I.flatten(), Y=J.flatten())

    da_test = em.sel2d(da, lon_rho=Lon, lat_rho=Lat).squeeze()

    assert np.allclose(da_check, da_test)


def test_index_reuse():
    """Test reusing calculated index."""

    da = model["da"].copy()  # need fresh one

    # no index ahead of time
    assert da.xoak.index is None

    t1temp = time()
    em.sel2d(da, lon_rho=lon, lat_rho=lat).squeeze()
    t1 = time() - t1temp

    assert da.xoak.index is not None

    t2temp = time()
    em.sel2d(da, lon_rho=lon, lat_rho=lat).squeeze()
    t2 = time() - t2temp

    if t2 >= t1:
        warnings.warn(
            "2D selection time did not improve after indexing, index may not be used.",
            RuntimeWarning,
        )


def test_ll_name_reversal():
    """Test reverse order of lonname, latname"""

    da1 = em.sel2d(da, lon_rho=lon, lat_rho=lat).squeeze()
    da2 = em.sel2d(da, lat_rho=lat, lon_rho=lon).squeeze()
    assert np.allclose(da1.to_array(), da2.to_array())


def test_sel_time():
    """Run with sel on time too."""

    # time is 0 for index and for datetime string
    da_check = da.cf.isel(X=i, Y=j, T=0)

    da_test = em.sel2d(da, lon_rho=lon, lat_rho=lat, ocean_time=0)

    assert np.allclose(da_check, da_test.to_array())

    # Won't run in different input order
    with pytest.raises(ValueError):
        em.sel2d(da, ocean_time=0, lon_rho=lon, lat_rho=lat)


def test_cf_versions():
    """Test cf name versions of everything"""

    da_check = em.sel2d(da, lon_rho=lon, lat_rho=lat)
    da_test = em.sel2dcf(da, longitude=lon, latitude=lat)
    assert np.allclose(da_check.to_array(), da_test.to_array())

    da_test = em.sel2dcf(da, latitude=lat, longitude=lon)
    assert np.allclose(da_check.to_array(), da_test.to_array())

    da_check = em.sel2d(da, lon_rho=lon, lat_rho=lat, ocean_time=0)
    da_test = em.sel2dcf(da, latitude=lat, longitude=lon, T=0)
    assert np.allclose(da_check.to_array(), da_test.to_array())

    da_test = em.sel2dcf(da, T=0, longitude=lon, latitude=lat)
    assert np.allclose(da_check.to_array(), da_test.to_array())
