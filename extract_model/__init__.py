"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

import ast  # noqa: F401

import cf_xarray as cfxr  # noqa: F401
import requests  # noqa: F401
from pkg_resources import DistributionNotFound, get_distribution

from .extract_model import make_output_ds, select  # noqa: F401

try:
    __version__ = get_distribution("extract_model").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
