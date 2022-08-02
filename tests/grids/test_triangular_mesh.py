#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Tests for triangular mesh stuff."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extract_model.grids.triangular_mesh import UnstructuredGridSubset


@pytest.fixture
def fake_fvcom() -> xr.Dataset:
    file_pth = Path(__file__).parent.parent / "data/fake_fvcom.nc"
    with xr.open_dataset(file_pth) as ds:
        yield ds


@pytest.fixture
def real_fvcom() -> xr.Dataset:
    file_pth = Path(__file__).parent.parent / "data/test_leofs_fvcom.nc"
    with xr.open_dataset(
        file_pth,
        engine="triangularmesh_netcdf",
        decode_times=True,
        preload_varmap={"siglay": "sigma_layers", "siglev": "sigma_levels"},
        drop_variables=["Itime", "Itime2"],
    ) as ds:
        yield ds


def test_triangle_algorithms(fake_fvcom):
    # The fake_fvcom file contains a known triangulation and the BBOX was carefully selected to
    # cover all cases of where triangles can exist in relation to the bbox:
    # - At least one point in the BBOX
    # - At least one point of the BBOX in the triangle
    # - An edge intersection
    bbox = [11.5, 2.5, 25, 13]
    subsetter = UnstructuredGridSubset()
    mask = subsetter.get_intersecting_mask(fake_fvcom, bbox, "fvcom")
    np.testing.assert_equal(
        np.where(mask)[0],
        np.array([3, 10, 12, 13, 14, 15, 54, 55, 57, 85, 87, 88]),
    )


def test_subset(real_fvcom):
    bbox = (276.4, 41.5, 277.4, 42.1)
    subsetter = UnstructuredGridSubset()
    ds = subsetter.subset(real_fvcom, bbox, "fvcom")
    assert ds is not None
    assert ds.dims["node"] == 1833
    assert ds.dims["nele"] == 3392
    # Check a node variable
    np.testing.assert_allclose(
        ds["x"][:10],
        np.array(
            [
                543232.0,
                544512.0,
                546048.0,
                547584.0,
                549056.0,
                544512.0,
                543232.0,
                545920.0,
                547584.0,
                549056.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(ds["nv"][:, 0], np.array([6, 7, 1], dtype=np.int32))


def test_subset_accessor(real_fvcom):
    bbox = (276.4, 41.5, 277.4, 42.1)
    ds = real_fvcom.em.sub_bbox(bbox)
    assert ds is not None
    assert ds.dims["node"] == 1833
    assert ds.dims["nele"] == 3392
    # Check a node variable
    np.testing.assert_allclose(
        ds["x"][:10],
        np.array(
            [
                543232.0,
                544512.0,
                546048.0,
                547584.0,
                549056.0,
                544512.0,
                543232.0,
                545920.0,
                547584.0,
                549056.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(ds["nv"][:, 0], np.array([6, 7, 1], dtype=np.int32))

    ds = real_fvcom.em.sub_bbox(bbox, model_type="FVCOM")
    assert ds is not None
    assert ds.dims["node"] == 1833
    assert ds.dims["nele"] == 3392


def test_sub_grid_accessor(real_fvcom):
    bbox = (276.4, 41.5, 277.4, 42.1)
    ds = real_fvcom.em.sub_grid(bbox=bbox)
    assert ds is not None
    assert ds.dims["node"] == 1833
    assert ds.dims["nele"] == 3392
    # Check a node variable
    np.testing.assert_allclose(
        ds["x"][:10],
        np.array(
            [
                543232.0,
                544512.0,
                546048.0,
                547584.0,
                549056.0,
                544512.0,
                543232.0,
                545920.0,
                547584.0,
                549056.0,
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(ds["nv"][:, 0], np.array([6, 7, 1], dtype=np.int32))

    ds = real_fvcom.em.sub_grid(bbox=bbox, model_type="FVCOM")
    assert ds is not None
    assert ds.dims["node"] == 1833
    assert ds.dims["nele"] == 3392


def test_filter(real_fvcom):

    standard_names = ["sea_water_temperature"]
    ds_filtered = real_fvcom.em.filter(standard_names=standard_names)
    varnames = sorted(ds_filtered.variables)
    for coord_var in UnstructuredGridSubset.FVCOM_COORDINATE_VARIABLES:
        assert coord_var in varnames

    for coord_var in ("x", "y", "xc", "yc"):
        assert coord_var in varnames
