#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class definition for the ModelTypes enum class."""
from enum import Enum


class ModelType(str, Enum):
    """Supported Models."""
    ROMS = 'ROMS'
    FVCOM = 'FVCOM'
    SELFE = 'SELFE'
    HYCOM = 'HYCOM'
    POM = 'POM'
    RTOFS = 'RTOFS'
    GFS = 'GFS'
    ADCIRC = 'ADCIRC'
