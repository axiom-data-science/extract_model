import extract_model as em
import numpy as np

from pathlib import Path
from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


def test_2dsel():
    """Test accessor sel2d and argsel2d

    indices saved from first use of sel2d."""

    model = models[0]
    da = model["da"]
    i, j = model["i"], model["j"]

    if da.cf["longitude"].ndim == 1:
        longitude = float(da.cf["X"][i])
        latitude = float(da.cf["Y"][j])

    elif da.cf["longitude"].ndim == 2:
        longitude = float(da.cf["longitude"][j, i])
        latitude = float(da.cf["latitude"][j, i])

    # take a nearby point to test function
    lon_comp = longitude + 0.001
    lat_comp = latitude + 0.001

    da_sel2d = da.em.sel2d(lon_comp, lat_comp)
    da_check = da.cf.isel(X=i, Y=j)

    # checks that the resultant model output is the same
    assert np.allclose(da_sel2d, da_check)

    # checks that the indices are the same
    assert (i,j) == da.em.argsel2d_map[(lon_comp, lat_comp)]
