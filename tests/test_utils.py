#!/usr/bin/env pytest
"""
Test utilities.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import extract_model as em

from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


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
    )
    da_compare = da.where(box, drop=True)
    ds_compare = ds.where(box, drop=True)

    assert np.allclose(da_out, da_compare, equal_nan=True)
    assert ds_out.equals(ds_compare)


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
