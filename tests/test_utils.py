"""
Test utilities.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import extract_model as em


# Test all models on the following
models = []

# MOM6 inputs
url = Path(__file__).parent / "test_mom6.nc"
var_name = "uo"
bbox = [-152, 54.0, -148, 56.0]
mom6 = dict(
    url=url,
    var_name=var_name,
    bbox=bbox,
)
models += [mom6]

# HYCOM
url = Path(__file__).parent / "test_hycom.nc"
var_name = "water_u"
bbox = [146, -14, 148, -12]
hycom = dict(
    url=url,
    var_name=var_name,
    bbox=bbox,
)
models += [hycom]

# HYCOM2
url = Path(__file__).parent / "test_hycom2.nc"
var_name = "u"
bbox = [-91.8, 28.0, -91.2, 29.0]
hycom2 = dict(
    url=url,
    var_name=var_name,
    bbox=bbox,
)
models += [hycom2]

# ROMS
url = Path(__file__).parent / "test_roms.nc"
var_name = "zeta"
bbox = [-92, 27, -91, 29]
roms = dict(
    url=url,
    var_name=var_name,
    bbox=bbox,
)
models += [roms]


@pytest.mark.parametrize("model", models)
class TestModel:
    def test_sub_bbox(self, model):
        """Test sub_bbox on DataArray and Dataset.

        This only works with the ROMS output because it is a narrow
        sample Dataset and has only one horizontal grid.
        """

        url, var_name, bbox = model["url"], model["var_name"], model["bbox"]

        # Dataset
        ds = xr.open_mfdataset([url], preprocess=em.preprocess)
        ds_out = ds.em.sub_bbox(bbox=bbox, drop=True)

        da = ds[var_name]
        da_out = da.em.sub_bbox(bbox=bbox, drop=True)

        box = (
            (bbox[0] < ds.cf["longitude"])
            & (ds.cf["longitude"] < bbox[2])
            & (bbox[1] < ds.cf["latitude"])
            & (ds.cf["latitude"] < bbox[3])
        )
        da_compare = da.where(box, drop=True)
        ds_compare = ds.where(box, drop=True)

        assert np.allclose(da_out, da_compare, equal_nan=True)
        assert ds_out.equals(ds_compare)

    def test_sub_grid_ds_roms(self, model):
        """Test subset on Dataset."""

        url, var_name, bbox = model["url"], model["var_name"], model["bbox"]

        # Dataset
        ds = xr.open_mfdataset([url], preprocess=em.preprocess)
        # bbox = [-92, 27, -91, 29]
        ds_out = ds.em.sub_grid(bbox=bbox)
        da_compare = ds[var_name].em.sub_bbox(bbox=bbox)

        X, Y = da_compare.cf["X"].values, da_compare.cf["Y"].values
        sel_dict = {"X": X, "Y": Y}
        ds_new = ds.cf.sel(sel_dict)

        assert ds_out.equals(ds_new)
