#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Tests for triangular mesh stuff."""
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pytest

from extract_model.grids.triangular_mesh import UnstructuredGridSubset


@pytest.fixture
def fake_fvcom():
    file_pth = Path(__file__).parent.parent / "data/fake_fvcom.nc"
    with nc4.Dataset(file_pth) as nc:
        yield nc


def test_triangle_algorithms(fake_fvcom):
    # The fake_fvcom file contains a known triangulation and the BBOX was carefully selected to
    # cover all cases of where triangles can exist in relation to the bbox:
    # - At least one point in the BBOX
    # - At least one point of the BBOX in the triangle
    # - An edge intersection
    bbox = [11.5, 2.5, 25, 13]
    subsetter = UnstructuredGridSubset()
    mask = subsetter.get_intersecting_mask(fake_fvcom, bbox)
    np.testing.assert_equal(
        np.where(mask)[0],
        np.array([3, 10, 12, 13, 14, 15, 54, 55, 57, 85, 87, 88]),
    )
