#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module to help generate fake data that looks like real model output."""
import os
import subprocess
import tempfile

from pathlib import Path
from typing import Generator, List

import xarray as xr


def get_offerings() -> List[str]:
    """Return a list of netCDF CDL templates offered."""
    return list((Path(__file__).parent / "data").glob("*.cdl"))


def create_model_tempfile(cdl_path: Path) -> Path:
    """Return path to temporary file (to be cleaned up by caller) with model data."""
    fd, pth = tempfile.mkstemp()
    os.close(fd)
    subprocess.run(["ncgen", "-o", pth, str(cdl_path)], check=True)
    return Path(pth)


def yield_from_cdl(model_name: str) -> Generator[xr.Dataset, None, None]:
    """Yield an open xr.Dataset and then clean up temporary files."""
    files = [f for f in get_offerings() if f.stem == model_name]
    if len(files) < 1:
        raise ValueError(f"Invalid model name, can't find CDL: {model_name}")
    cdl_file = files[0]
    pth = create_model_tempfile(cdl_file)
    try:
        yield xr.open_dataset(pth)
    finally:
        if pth.exists():
            pth.unlink()
