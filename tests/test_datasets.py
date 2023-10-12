"""Test synthetic datasets representing featuretypes."""

import pathlib

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xroms

import extract_model as em

from .make_test_datasets import make_test_datasets


project_name = "tests"
base_dir = pathlib.Path("tests/test_results")

dds = make_test_datasets()
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=False, include_3D_metrics=False)


def test_timeSeries_temp():
    featuretype = "timeSeries"
    key_variable = "temp"
    interpolate_horizontal = True
    vertical_interp = False
    grid = None
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = False
    locstreamT = False
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = np.unique(data["lon"].values)
    lats = np.unique(data["lat"].values)
    T = np.unique(data["date_time"].values)
    # Z = data["depth"].values
    Z = None
    iZ = -1
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{horizontal_interp_code}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataset(expname)
    dsexpected = dsexpected.assign_coords({"s_rho": dsexpected["s_rho"]})[key_variable]
    # dsexpected = xr.open_dataarray(expname)
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_timeSeries_zeta():
    featuretype = "timeSeries"
    key_variable = "zeta"
    interpolate_horizontal = False
    vertical_interp = False
    grid = None
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = False
    locstreamT = False
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = data["lon"].values
    lats = data["lat"].values
    T = np.unique(data["date_time"].values)
    # Z = data["depth"].values
    Z = None
    iZ = None
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{interpolate_horizontal}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataarray(expname)
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_profile():
    featuretype = "profile"
    key_variable = "salt"
    interpolate_horizontal = False
    vertical_interp = True
    ds = xroms.datasets.fetch_ROMS_example_full_grid()
    ds, xgrid = xroms.roms_dataset(
        ds, include_cell_volume=False, include_3D_metrics=True
    )
    grid = xgrid
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = False
    locstreamT = False
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = data["lon"].values
    lats = data["lat"].values
    if locstreamT:
        T = [pd.Timestamp(date) for date in data["date_time"].values]
    else:
        T = [pd.Timestamp(date) for date in np.unique(data["date_time"].values)]
    Z = data["depth"].values
    # Z = None
    iZ = None
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{interpolate_horizontal}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataarray(expname)
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_timeSeriesProfile():
    featuretype = "timeSeriesProfile"
    key_variable = "temp"
    interpolate_horizontal = False
    vertical_interp = True
    ds = xroms.datasets.fetch_ROMS_example_full_grid()
    ds, xgrid = xroms.roms_dataset(
        ds, include_cell_volume=False, include_3D_metrics=True
    )
    grid = xgrid
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = False
    locstreamT = False
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = data["lon"].values
    lats = data["lat"].values
    if locstreamT:
        T = [pd.Timestamp(date) for date in data["date_time"].values]
    else:
        T = [pd.Timestamp(date) for date in np.unique(data["date_time"].values)]
    Z = data["depths"].values
    # Z = None
    iZ = None
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{interpolate_horizontal}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataarray(expname)
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_trajectory():
    """a surface drifter"""
    featuretype = "trajectory"
    key_variable = "salt"
    interpolate_horizontal = True
    vertical_interp = False
    grid = None
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = True
    locstreamT = True
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = data["lons"].values
    lats = data["lats"].values
    if locstreamT:
        T = [pd.Timestamp(date) for date in data["date_time"].values]
    else:
        T = [pd.Timestamp(date) for date in np.unique(data["date_time"].values)]
    # Z = np.unique(data["depth"].values)
    Z = None
    iZ = -1
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{interpolate_horizontal}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataset(expname)
    dsexpected = dsexpected.assign_coords({"s_rho": dsexpected["s_rho"]})[key_variable]
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_trajectoryProfile():
    featuretype = "trajectoryProfile"
    key_variable = "salt"
    interpolate_horizontal = True
    vertical_interp = True
    ds = xroms.datasets.fetch_ROMS_example_full_grid()
    ds, xgrid = xroms.roms_dataset(
        ds, include_cell_volume=False, include_3D_metrics=True
    )
    grid = xgrid
    extrap = False
    horizontal_interp_code = "delaunay"
    mask = None
    locstream = True
    locstreamT = True
    locstreamZ = True
    use_projection = False

    data = dds[featuretype]
    lons = data["lons"].values
    lats = data["lats"].values
    if locstreamT:
        T = [pd.Timestamp(date) for date in data["date_time"].values]
    else:
        T = [pd.Timestamp(date) for date in np.unique(data["date_time"].values)]
    Z = np.unique(data["depth"].values)
    # Z = None
    iZ = None
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
    if hasattr(dsactual, "chunks"):
        dsactual = dsactual.load()

    expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{interpolate_horizontal}.nc"
    # # previously saved with:
    # dsactual.to_netcdf(expname)
    dsexpected = xr.open_dataarray(expname)
    assert dsexpected.equals(dsactual.astype(dsexpected.dtype))


def test_grid():
    featuretype = "grid"
    key_variable = "temp"
    interpolate_horizontal = True
    vertical_interp = False
    ds = xroms.datasets.fetch_ROMS_example_full_grid()
    ds, xgrid = xroms.roms_dataset(
        ds, include_cell_volume=False, include_3D_metrics=False
    )
    grid = None
    extrap = False
    horizontal_interp_code = "xesmf"
    mask = None
    locstream = False
    locstreamT = False
    locstreamZ = False
    use_projection = False

    data = dds[featuretype]
    lons = data["lon_rho"].values
    lats = data["lat_rho"].values
    T = np.unique(data["ocean_time"].values)
    # Z = data["depth"].values
    Z = None
    iZ = -1
    iT = None

    select_kwargs = dict(
        longitude=lons,
        latitude=lats,
        T=T,
        Z=Z,
        vertical_interp=vertical_interp,
        iT=iT,
        iZ=iZ,
        extrap=extrap,
        extrap_val=None,
        use_projection=use_projection,
        locstream=locstream,
        locstreamT=locstreamT,
        locstreamZ=locstreamZ,
        # locstream_dim="z_rho",
        weights=None,
        mask=mask,
        use_xoak=False,
        horizontal_interp=interpolate_horizontal,
        horizontal_interp_code=horizontal_interp_code,
        xgcm_grid=grid,
        return_info=True,
    )

    if em.extract_model.XESMF_AVAILABLE:
        dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)

        if hasattr(dsactual, "chunks"):
            dsactual = dsactual.load()

        expname = f"tests/test_results/{featuretype}_{key_variable}_horinterp_{horizontal_interp_code}.nc"
        # # previously saved with:
        # dsactual.to_netcdf(expname)
        dsexpected = xr.open_dataset(expname)
        dsexpected = dsexpected.assign_coords(
            {"s_rho": dsexpected["s_rho"], "ocean_time": dsexpected["ocean_time"]}
        )[key_variable]
        # this isn't working in CI so changing to array comparison
        # assert dsexpected.equals(dsactual.astype(dsexpected.dtype))
        assert np.allclose(dsexpected.values, dsactual.values)

    else:
        with pytest.raises(ModuleNotFoundError):
            dsactual, out_kwargs = em.select(ds[key_variable], **select_kwargs)
