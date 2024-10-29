#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Tests for triangular mesh stuff."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extract_model import preprocessing, utils
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
        chunks={"time": 1},
    ) as ds:
        yield ds


@pytest.fixture
def selfe_data() -> xr.Dataset:
    """Fixture for CREOFS SELFE data."""
    file_pth = Path(__file__).parent.parent / "data/test_creofs_selfe.nc"
    with xr.open_dataset(
        file_pth,
        decode_times=True,
        chunks={"time": 1},
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


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_fvcom_subset(real_fvcom, preload):
    bbox = (276.4, 41.5, 277.4, 42.1)
    subsetter = UnstructuredGridSubset()
    ds = subsetter.subset(real_fvcom, bbox, "fvcom", preload=preload)
    assert ds is not None
    assert ds.sizes["node"] == 1833
    assert ds.sizes["nele"] == 3392
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
    assert not np.any(ds["nv"][:] < 1)


def test_fvcom_subset_accessor(real_fvcom):
    bbox = (276.4, 41.5, 277.4, 42.1)
    ds = real_fvcom.em.sub_bbox(bbox)
    assert ds is not None
    assert ds.sizes["node"] == 1833
    assert ds.sizes["nele"] == 3392
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
    assert ds.sizes["node"] == 1833
    assert ds.sizes["nele"] == 3392


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_fvcom_sub_grid_accessor(real_fvcom, preload):
    bbox = (276.4, 41.5, 277.4, 42.1)
    ds = real_fvcom.em.sub_grid(bbox=bbox, preload=preload)
    assert ds is not None
    assert ds.sizes["node"] == 1833
    assert ds.sizes["nele"] == 3392
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

    ds = real_fvcom.em.sub_grid(bbox=bbox, model_type="FVCOM", preload=preload)
    assert ds is not None
    assert ds.sizes["node"] == 1833
    assert ds.sizes["nele"] == 3392


def test_fvcom_filter(real_fvcom):
    real_fvcom["sigma_layers"].attrs[
        "formula_terms"
    ] = "sigma: sigma_layers eta: zeta depth: h"
    real_fvcom["sigma_levels"].attrs[
        "formula_terms"
    ] = "sigma: sigma_levels eta: zeta depth: h"
    real_fvcom = real_fvcom.assign_coords(
        {
            "time": real_fvcom["time"],
            "sigma_layers": real_fvcom["sigma_layers"],
            "sigma_levels": real_fvcom["sigma_levels"],
            "lat": real_fvcom["lat"],
            "lon": real_fvcom["lon"],
            "latc": real_fvcom["latc"],
            "lonc": real_fvcom["lonc"],
        }
    )

    standard_names = ["sea_water_temperature"]
    ds_filtered = real_fvcom.em.filter(standard_names=standard_names)
    varnames = sorted(ds_filtered.variables)
    for coord_var in UnstructuredGridSubset.FVCOM_COORDINATE_VARIABLES:
        assert coord_var in varnames

    for coord_var in ("x", "y", "xc", "yc"):
        assert coord_var in varnames


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_fvcom_subset_scalars(real_fvcom, preload):
    bbox = (276.4, 41.5, 277.4, 42.1)
    xvar = xr.DataArray(data=np.array(0.0), attrs={"long_name": "Example Data"})
    ds = real_fvcom.assign(variables={"example": xvar})
    ds_ss = ds.em.sub_grid(bbox=bbox, preload=preload)
    assert ds_ss is not None
    assert ds_ss.sizes["node"] == 1833
    assert ds_ss.sizes["nele"] == 3392
    assert "example" in ds_ss.variables
    assert len(ds_ss["example"].sizes) < 1


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_fvcom_nv_reindexing(real_fvcom, preload):
    bbox = (280, 42.2, 281, 43)
    ds_ss = real_fvcom.em.sub_grid(bbox=bbox, preload=preload)
    expected = np.array([[1, 11, 11], [10, 10, 1], [9, 1, 2]])
    np.testing.assert_equal(expected, ds_ss["nv"].to_numpy()[:, :3])


def test_fvcom_preload(real_fvcom):
    """Test preloading vs normal reindexing."""
    bbox = (280, 42.2, 281, 43)
    ds_ss_norm = real_fvcom.em.sub_grid(bbox=bbox, preload=False)
    ds_ss_preload = real_fvcom.em.sub_grid(bbox=bbox, preload=True)
    for variable in sorted(ds_ss_norm.variables):
        raw_data_norm = ds_ss_norm[variable].to_numpy()
        raw_data_preload = ds_ss_preload[variable].to_numpy()
        np.testing.assert_equal(raw_data_norm, raw_data_preload)


def test_fvcom_preprocess(real_fvcom):
    ds = preprocessing.preprocess(real_fvcom)
    assert ds is not None


def test_selfe_sub_bbox_accessor(selfe_data):
    bbox = (-123.8, 46.2, -123.6, 46.3)
    ds_ss = selfe_data.em.sub_bbox(bbox=bbox)
    assert ds_ss is not None
    assert ds_ss.sizes["node"] == 4273
    assert ds_ss.sizes["nele"] == 8178
    np.testing.assert_allclose(
        ds_ss["x"][:10],
        np.array(
            [
                370944.0,
                370944.0,
                370752.0,
                370688.0,
                370624.0,
                370688.0,
                370880.0,
                370880.0,
                370816.0,
                370816.0,
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_selfe_sub_grid_accessor(selfe_data, preload):
    bbox = (-123.8, 46.2, -123.6, 46.3)
    ds_ss = selfe_data.em.sub_grid(bbox=bbox, preload=preload)
    assert ds_ss is not None
    assert ds_ss.sizes["node"] == 4273
    assert ds_ss.sizes["nele"] == 8178
    np.testing.assert_allclose(
        ds_ss["x"][:10],
        np.array(
            [
                370944.0,
                370944.0,
                370752.0,
                370688.0,
                370624.0,
                370688.0,
                370880.0,
                370880.0,
                370816.0,
                370816.0,
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("preload", [False, True], ids=lambda x: f"preload={x}")
def test_selfe_subset_scalars(selfe_data, preload):
    xvar = xr.DataArray(data=np.array(0.0), attrs={"long_name": "Example Data"})
    ds = selfe_data.assign(variables={"example": xvar})
    bbox = (-123.8, 46.2, -123.6, 46.3)
    ds_ss = ds.em.sub_grid(bbox=bbox, preload=preload)
    assert ds_ss is not None
    assert ds_ss.sizes["node"] == 4273
    assert ds_ss.sizes["nele"] == 8178
    assert "example" in ds_ss.variables
    assert len(ds_ss["example"].sizes) < 1


def test_selfe_preload(selfe_data: xr.Dataset):
    """Test preloading vs normal reindexing."""
    bbox = (-123.8, 46.2, -123.6, 46.3)
    ds_ss_norm = selfe_data.em.sub_grid(bbox=bbox, preload=False)
    ds_ss_preload = selfe_data.em.sub_grid(bbox=bbox, preload=True)
    for variable in sorted(ds_ss_norm.variables):
        raw_data_norm = ds_ss_norm[variable].to_numpy()
        raw_data_preload = ds_ss_preload[variable].to_numpy()
        np.testing.assert_equal(raw_data_norm, raw_data_preload)


def test_selfe_filter(selfe_data):
    standard_names = ["sea_water_temperature"]
    filtered_ds = selfe_data.em.filter(standard_names=standard_names)
    assert "temp" in filtered_ds.variables


def test_selfe_preprocess(selfe_data):
    ds = preprocessing.preprocess(selfe_data)
    assert ds is not None


def test_unsupported_grid(selfe_data):
    subsetter = UnstructuredGridSubset()
    with pytest.raises(ValueError):
        subsetter.subset(None, (0, 0, 1, 1), "unreal")
    with pytest.raises(ValueError):
        subsetter.get_intersecting_mask(None, (0, 0, 1, 1), "unreal")


def test_non_intersecting_subset(real_fvcom, selfe_data):
    bbox = (0, 0, 10, 10)
    with pytest.raises(ValueError):
        real_fvcom.em.sub_grid(bbox=bbox)
    with pytest.raises(ValueError):
        selfe_data.em.sub_grid(bbox=bbox)
