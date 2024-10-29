#!/usr/bin/env pytest
"""
Test utilities.
"""

import subprocess

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import extract_model as em

from extract_model.utils import sub_grid

from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    dirpth = tmp_path_factory.mktemp("data")
    yield dirpth


@pytest.fixture(scope="module", params=["ciofs", "tbofs", "cbofs", "dbofs"])
def roms_model(request, data_dir):
    tmp_pth = data_dir / f"{request.param}.nc"
    pth = Path(__file__).parent / f"dynamic/data/{request.param}.cdl"
    subprocess.run(["ncgen", "-o", str(tmp_pth), str(pth)], check=True)
    ds = xr.open_dataset(tmp_pth)
    return ds


def test_roms_coordinate_filtering(roms_model: xr.Dataset):
    """Test em.filter to ensure ROMS coordinates are maintained."""
    standard_names = ["sea_water_temperature"]
    ds = roms_model.em.filter(standard_names)
    coords = ("lon", "lat")
    roms_coords = ("rho", "psi", "u", "v")
    for coord, roms_coord in product(coords, roms_coords):
        coord_name = f"{coord}_{roms_coord}"
        assert coord_name in ds.coords

    ds = roms_model.em.filter(standard_names, keep_horizontal_coords=False)
    # Make sure that only the rho coordinates remain
    roms_coords = ("psi", "u", "v")
    for coord, roms_coord in product(coords, roms_coords):
        coord_name = f"{coord}_{roms_coord}"
        assert coord_name not in ds.coords


def test_roms_mask_filtering(roms_model: xr.Dataset):
    """Test em.filter to ensure ROMS mask variables are maintained."""
    standard_names = ["sea_water_temperature"]
    ds = roms_model.em.filter(standard_names)

    has_wetdry_mask = any((i.startswith("wetdry_mask") for i in roms_model.variables))
    mask_names = ["mask"]
    if has_wetdry_mask:
        mask_names.append("wetdry_mask")

    roms_coords = ("rho", "psi", "u", "v")
    for mask_name, coord in product(mask_names, roms_coords):
        coord_name = f"{mask_name}_{coord}"
        # Masks are not actually coordinates
        assert coord_name in ds.variables
    ds = roms_model.em.filter(standard_names, keep_coord_mask=False)
    roms_coords = ("rho", "psi", "u", "v")
    for mask_name, coord in product(mask_names, roms_coords):
        coord_name = f"{mask_name}_{coord}"
        # Masks are not actually coordinates
        assert coord_name not in ds.variables


def test_roms_sub_grid():
    """Test a specific case for sub_grid on ROMS from NOAA OFS."""
    # This dataset is a manually selected region of the ROMS dataset with just enough metadata to
    # describe the horizontal and vertical coordinates.
    pth = Path(__file__).parent / "data/dbofs_sample.nc"
    bbox = (-75.0, 38.0, -74.0, 38.4)
    ds = xr.open_mfdataset([pth], preprocess=em.preprocess)
    sub_ds = ds.em.sub_grid(bbox)
    assert sub_ds.coords["lon_rho"].shape == (37, 28)
    assert np.abs((sub_ds["lon_rho"].min().values - -75.02909851074219)) < 0.00001

    # Check that we can sub-grid if xi_rho and eta_rho aren't coordinate variables
    ds = xr.open_mfdataset([pth], preprocess=em.preprocess)
    del ds.coords["xi_rho"]
    del ds.coords["eta_rho"]
    # Will raise if it can't sub-grid
    ds.em.sub_grid(bbox)


@pytest.mark.parametrize("model", models, ids=lambda x: x["name"])
def test_sub_bbox(model):
    """Test sub_bbox on DataArray and Dataset.

    This only works with the ROMS output because it is a narrow
    sample Dataset and has only one horizontal grid.
    """

    var_name, bbox = model["var"], model["sub_bbox"]
    pth = eval(model["url"])

    # Dataset
    ds = xr.open_mfdataset([pth], preprocess=em.preprocess)
    ds_out = ds.em.sub_bbox(bbox=bbox, drop=True)

    da = ds[var_name]
    da_out = da.em.sub_bbox(bbox=bbox, drop=True)

    box = (
        (bbox[0] < ds.cf["longitude"])
        & (ds.cf["longitude"] < bbox[2])
        & (bbox[1] < ds.cf["latitude"])
        & (ds.cf["latitude"] < bbox[3])
    ).load()
    da_compare = da.where(box, drop=True)
    ds_compare = ds.where(box, drop=True)

    assert np.allclose(da_out, da_compare, equal_nan=True)
    assert ds_out.equals(ds_compare)


@pytest.mark.parametrize("model", models, ids=lambda x: x["name"])
def test_naive_sub_bbox(model):
    if model["name"] == "MOM6":
        # MOM6 doesn't provide a CF-compliant grid description, clients
        # shouldn't use naive_subbox for MOM6.
        pytest.skip("MOM6 is not supported by naive_subbox")
        return
    var_name, bbox = model["var"], model["sub_bbox"]
    pth = eval(model["url"])

    # Dataset
    ds = xr.open_mfdataset([pth], preprocess=em.preprocess)
    ds_out = ds.em.sub_grid(bbox=bbox, naive=True)
    for dim, value in model["naive_subbox"].items():
        assert ds_out.sizes[dim] == value


@pytest.mark.parametrize("model", models, ids=lambda x: x["name"])
def test_sub_grid_ds(model):
    """Test subset on Dataset."""

    var_name, bbox = model["var"], model["sub_bbox"]
    url = eval(model["url"])

    # Dataset
    ds = xr.open_mfdataset([url], preprocess=em.preprocess)
    # if 'roms' in url.stem:
    #     import pdb; pdb.set_trace()
    ds_out = ds.em.sub_grid(bbox=bbox)
    if "roms" not in url.stem:

        da_compare = ds[var_name].em.sub_bbox(bbox=bbox)

        X, Y = da_compare.cf["X"].values, da_compare.cf["Y"].values
        sel_dict = {"X": X, "Y": Y}
        ds_new = ds.cf.sel(sel_dict)

        assert ds_out.equals(ds_new)


def test_sub_grid_assert():
    """Ensures that sub_grid will raise if passed an xr.DataArray."""
    xvar = xr.DataArray(data=np.arange(20), dims=("time",))
    bbox = (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    with pytest.raises(ValueError):
        sub_grid(ds=xvar, bbox=bbox)


def test_adding_axis_Z():
    """Check for specific case where Z is added in preprocessing."""

    ds = xr.Dataset()
    attrs = {
        "standard_name": "depth",
        "units": "m",
        "long_name": "Depth",
        "positive": "down",
    }
    ds["depth"] = ("depth", np.arange(10), attrs)

    ds = em.preprocess(ds)
    assert "Z" in ds.cf.axes


def test_naive_subbox_illegal_grid():
    """Ensure that we raise when the grid is invalid.

    An invalid grid can sometimes arise from someone mislabling a variable as a
    coordinate or other copypasta errors.
    """
    eta = np.arange(100)
    xi = np.arange(100)
    lon = np.linspace(-30, 40, 100)
    _, lat = np.meshgrid(lon, np.linspace(-20, 20, 100))
    data_dict = {}
    data_dict["eta"] = xr.DataArray(eta, dims=("eta",))
    data_dict["xi"] = xr.DataArray(xi, dims=("xi",))
    data_dict["lon"] = xr.DataArray(
        lon,
        dims=("lon",),
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    data_dict["lat"] = xr.DataArray(
        lat,
        dims=("eta", "xi"),
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    ds = xr.Dataset(data_dict)
    with pytest.raises(ValueError) as err:
        em.utils.naive_subbox(ds=ds, bbox=(0, 0, 5, 5))
        assert str(err) == "Invalid grid detected"


def test_filter_with_angle():
    """Ensure that em.filter will keep an angle variable even without an appropriate standard_name."""
    eta = np.arange(40)
    xi = np.arange(20)
    angle = np.arange(40 * 20).reshape(40, 20)
    data_dict = {}
    data_dict["eta"] = xr.DataArray(eta, dims=("eta",))
    data_dict["xi"] = xr.DataArray(xi, dims=("xi",))
    data_dict["angle"] = xr.DataArray(angle, dims=("eta", "xi"))
    ds = xr.Dataset(data_dict)
    ds = ds.em.filter([])
    assert "angle" in ds.variables
