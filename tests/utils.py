from pathlib import Path  # noqa E401

import xarray as xr
import yaml


def read_model_configs(model_configs_file):
    """Read model configs from file and return as a list of dicts."""

    with open(model_configs_file) as f:
        configs = yaml.safe_load(f)

    for _, config in configs.items():
        path = eval(config["url"])
        with xr.open_dataset(path) as ds:
            ds = ds.cf.guess_coord_axis()
            da = ds[config['var']]
        config["da"] = da

        config["lonslice"] = eval(config["lonslice"])
        config["latslice"] = eval(config["latslice"])

    return [config for _, config in configs.items()]
