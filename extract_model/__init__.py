"""
Select model output using xarray and `isel` or interpolation through xarray or xesmf.
"""

import ast

import cf_xarray as cfxr
import requests

from pkg_resources import DistributionNotFound, get_distribution

from .extract_model import select


# For variable identification with cf-xarray
# custom_criteria to identify variables is saved here
# https://gist.github.com/kthyng/c3cc27de6b4449e1776ce79215d5e732
my_custom_criteria_gist = "https://gist.githubusercontent.com/kthyng/c3cc27de6b4449e1776ce79215d5e732/raw/55317e92be367f7d6d66e6142d5219a5d272afce/my_custom_criteria.py"
response = requests.get(my_custom_criteria_gist)
my_custom_criteria = ast.literal_eval(response.text)
cfxr.set_options(my_custom_criteria)


try:
    __version__ = get_distribution("extract_model").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
