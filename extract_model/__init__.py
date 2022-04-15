"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

import ast  # noqa: F401

import cf_xarray as cfxr  # noqa: F401
import requests  # noqa: F401
from pkg_resources import DistributionNotFound, get_distribution

import extract_model.accessor  # noqa: F401

from .extract_model import (argsel2d, make_output_ds, sel2d,  # noqa: F401
                            select)
from .pyinterp_shim import PyInterpShim  # noqa: F401
from .utils import filter, order, preprocess, sub_bbox, sub_grid  # noqa: F401

try:
    __version__ = get_distribution("extract_model").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
