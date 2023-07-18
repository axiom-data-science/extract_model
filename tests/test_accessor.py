from pathlib import Path

import numpy as np

import extract_model as em

from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


def test_2dsel():
    """Test accessor sel2d

    indices saved from first use of sel2d."""

    model = models[3]
    da = model["da"]
    i, j = model["i"], model["j"]
    varname = da.name

    if da.cf["longitude"].ndim == 1:
        longitude = float(da.cf["X"][i])
        latitude = float(da.cf["Y"][j])

    elif da.cf["longitude"].ndim == 2:
        longitude = float(da.cf["longitude"][j, i])
        latitude = float(da.cf["latitude"][j, i])

    # take a nearby point to test function
    lon_comp = longitude + 0.001
    lat_comp = latitude + 0.001

    inputs = {
        da.cf["longitude"].name: lon_comp,
        da.cf["latitude"].name: lat_comp,
        # "distances_name": "distance",
    }
    da_sel2d, kwargs_out_sel2d_acc_check = da.em.sel2d(**inputs, return_info=True)
    da_check = da.cf.isel(X=i, Y=j)
    da_sel2d_check = da_sel2d[varname]

    # checks that the resultant model output is the same
    assert np.allclose(da_sel2d_check.squeeze(), da_check)

    da_test, kwargs_out = da.em.sel2dcf(
        longitude=lon_comp, latitude=lat_comp, return_info=True, #distances_name="distance"
    )
    assert np.allclose(da_sel2d[varname], da_test[varname])
    assert np.allclose(kwargs_out_sel2d_acc_check["distance"], kwargs_out["distance"])
