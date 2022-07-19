import extract_model as em
import numpy as np

from pathlib import Path
from .utils import read_model_configs


model_config_path = Path(__file__).parent / "model_configs.yaml"
models = read_model_configs(model_config_path)


def test_2dsel():
    """Test accessor sel2d

    indices saved from first use of sel2d."""

    model = models[3]
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

    inputs = {da.cf["longitude"].name: lon_comp,
              da.cf["latitude"].name: lat_comp}
    da_sel2d = da.em.sel2d(**inputs)
    da_check = da.cf.isel(X=i, Y=j)

    # checks that the resultant model output is the same
    assert np.allclose(da_sel2d.squeeze(), da_check)

    da_test = da.em.sel2dcf(longitude=lon_comp, latitude=lat_comp)
    assert np.allclose(da_sel2d, da_test)
