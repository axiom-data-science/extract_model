"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

import cf_xarray as cfxr  # noqa: F401

from importlib.metadata import version, PackageNotFoundError

import extract_model.accessor  # noqa: F401

from .extract_model import sel2d, sel2dcf, select, selZ  # noqa: F401
from .utils import filter, order, preprocess, sub_bbox, sub_grid  # noqa: F401


try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass
