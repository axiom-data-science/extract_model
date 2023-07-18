"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

from importlib.metadata import PackageNotFoundError, version

import cf_xarray as cfxr  # noqa: F401

import extract_model.accessor  # noqa: F401

from .extract_model import sel2d, sel2dcf, select, selZ  # noqa: F401
from .utils import filter, order, sub_bbox, sub_grid, guess_model_type  # noqa: F401
from .preprocessing import preprocess


try:
    __version__ = version("extract_model")
except PackageNotFoundError:
    # package is not installed
    pass
