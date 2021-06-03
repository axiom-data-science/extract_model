"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

from pkg_resources import DistributionNotFound, get_distribution
try:
    __version__ = get_distribution("cf_xarray").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

from .extract_model import get_var_cf, select
