from pathlib import Path  # noqa E401

import extract_model as em
import xarray as xr
import yaml

model_configs_file = 'model_configs.yaml'


def read_model_configs(model_configs_file):
    """Read model configs from file and return as a list of dicts."""

    with open(model_configs_file) as f:
        configs = yaml.safe_load(f)

    for _, config in configs.items():
        path = eval(config["url"])
        with xr.open_mfdataset([path], preprocess=em.preprocess) as ds:
            da = ds[config['var']]
        config["da"] = da

        config["lonslice"] = eval(config["lonslice"])
        config["latslice"] = eval(config["latslice"])

    return [config for _, config in configs.items()]
