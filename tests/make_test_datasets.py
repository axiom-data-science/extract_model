import numpy as np
import pandas as pd
import xarray as xr
import xroms


def make_test_datasets():
    # use example model output from xroms to make datasets
    ds = xroms.datasets.fetch_ROMS_example_full_grid()
    ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)

    dds = {}

    # time series
    example_loc = ds.isel(eta_rho=20, xi_rho=10, s_rho=-1)
    times = pd.date_range(
        str(example_loc.ocean_time.values[0]),
        str(example_loc.ocean_time.values[1]),
        freq="1H",
    )
    npts = len(times)
    df = pd.DataFrame(
        {
            "date_time": times,
            "depth": np.zeros(npts),
            "lon": np.ones(npts) * float(example_loc.lon_rho) + 0.01,
            "lat": np.ones(npts) * float(example_loc.lat_rho) + 0.01,
            "sea_surface_height": np.ones(npts) * float(example_loc["zeta"].mean()),
            "temperature": np.ones(npts) * float(example_loc["temp"].mean()),
            "salinity": np.ones(npts) * float(example_loc["salt"].mean()),
            # "sea_level": np.random.normal(float(example_loc["zeta"].mean()), size=npts),
            # "temperature": np.random.normal(float(example_loc["temp"].mean()), size=npts),
            # "salinity": np.random.normal(float(example_loc["salt"].mean()), size=npts),
        }
    )
    dds["timeSeries"] = df

    # CTD profile
    # negative depths
    example_loc = ds.sel(eta_rho=20, xi_rho=10)
    npts = 50
    df = pd.DataFrame(
        {
            "date_time": "2009-11-19T14:00",
            "depth": np.linspace(0, float(example_loc.z_rho[0, :].min()), npts),
            "lon": float(example_loc.lon_rho) + 0.01,
            "lat": float(example_loc.lat_rho) + 0.01,
            "temperature": np.linspace(
                float(example_loc["temp"].max()), float(example_loc["temp"].min()), npts
            ),
            "salinity": np.linspace(
                float(example_loc["salt"].min()), float(example_loc["salt"].max()), npts
            ),
        }
    )
    dds["profile"] = df

    ## surface drifter ##
    example_loc1 = ds.sel(eta_rho=20, xi_rho=10).isel(s_rho=-1)
    example_loc2 = ds.sel(eta_rho=20, xi_rho=15).isel(s_rho=-1)
    # positive depths

    times = pd.date_range(
        str(example_loc.ocean_time.values[0]),
        str(example_loc.ocean_time.values[1]),
        freq="1H",
    )
    npts = len(times)

    lons = np.linspace(float(example_loc1.lon_rho), float(example_loc2.lon_rho), npts)
    lats = [float(example_loc1.lat_rho)] * npts
    depths = [0] * npts

    # this is ready for per-data-point info now
    df = pd.DataFrame(index=times, data=dict(lons=lons, lats=lats))
    df.index.name = "date_time"
    df = df.reset_index()
    temp = np.linspace(
        float(example_loc1["temp"].max()),
        float(example_loc2["temp"].min()),
        npts,
    )
    salt = np.linspace(
        float(example_loc1["salt"].max()),
        float(example_loc2["salt"].min()),
        npts,
    )

    df["depth"] = depths
    df["temperature"] = temp
    df["salinity"] = salt
    dds["trajectory"] = df

    ## CTD transect ##
    example_loc1 = ds.sel(eta_rho=20, xi_rho=10)
    example_loc2 = ds.sel(eta_rho=20, xi_rho=15)
    # positive depths
    nstations = 5
    nptsperstation = 10
    depths = np.hstack(
        (
            np.linspace(0, abs(float(example_loc1.z_rho[0, :].min())), nptsperstation),
            np.linspace(0, abs(float(example_loc1.z_rho[0, :].min())), nptsperstation),
            np.linspace(0, abs(float(example_loc2.z_rho[0, :].min())), nptsperstation),
            np.linspace(0, abs(float(example_loc2.z_rho[0, :].min())), nptsperstation),
            np.linspace(0, abs(float(example_loc2.z_rho[0, :].min())), nptsperstation),
        )
    )
    # per station
    times = pd.date_range(
        str(example_loc.ocean_time.values[0]),
        str(example_loc.ocean_time.values[1]),
        freq="1H",
    )
    # repeats for each data points
    times_full = np.hstack(
        (
            [times[0]] * nptsperstation,
            [times[1]] * nptsperstation,
            [times[2]] * nptsperstation,
            [times[3]] * nptsperstation,
            [times[4]] * nptsperstation,
        )
    )
    lons = np.linspace(
        float(example_loc1.lon_rho), float(example_loc2.lon_rho), nstations
    )
    lats = [float(example_loc1.lat_rho)] * nstations
    # this is ready for per-data-point info now
    df = pd.DataFrame(index=times, data=dict(lons=lons, lats=lats)).reindex(times_full)
    df.index.name = "date_time"
    df = df.reset_index()
    temp1 = np.linspace(
        float(example_loc1["temp"].max()),
        float(example_loc1["temp"].min()),
        nptsperstation,
    )
    temp2 = np.linspace(
        float(example_loc2["temp"].max()),
        float(example_loc2["temp"].min()),
        nptsperstation,
    )
    temp = np.hstack(
        (
            temp1,
            temp1,
            temp1,
            temp2,
            # np.random.normal(temp1.mean(), size=nptsperstation),
            # np.random.normal(temp1.mean(), size=nptsperstation),
            # np.random.normal(temp2.mean(), size=nptsperstation),
            temp2,
        )
    )
    salt1 = np.linspace(
        float(example_loc1["salt"].min()),
        float(example_loc1["salt"].max()),
        nptsperstation,
    )
    salt2 = np.linspace(
        float(example_loc2["salt"].min()),
        float(example_loc2["salt"].max()),
        nptsperstation,
    )
    salt = np.hstack(
        (
            salt1,
            salt1,
            salt1,
            salt2,
            # np.random.normal(salt1.mean(), size=nptsperstation),
            # np.random.normal(salt1.mean(), size=nptsperstation),
            # np.random.normal(salt2.mean(), size=nptsperstation),
            salt2,
        )
    )

    df["depth"] = depths
    df["temperature"] = temp
    df["salinity"] = salt
    dds["trajectoryProfile"] = df

    # ADCP mooring
    example_loc = ds.sel(eta_rho=20, xi_rho=10)
    times = pd.date_range(
        str(example_loc.ocean_time.values[0]),
        str(example_loc.ocean_time.values[1]),
        freq="1H",
    )
    ntimes = len(times)
    ndepths = 20
    depths = np.linspace(0, float(example_loc.z_rho[0, :].min()), ndepths)
    lon = float(example_loc.lon_rho) + 0.01
    lat = float(example_loc.lat_rho) + 0.01
    temptemp = np.linspace(
        float(example_loc["temp"].max()), float(example_loc["temp"].min()), ndepths
    )
    temp = np.tile(temptemp[:, np.newaxis], [1, ntimes])
    salttemp = np.linspace(
        float(example_loc["salt"].min()), float(example_loc["salt"].max()), ndepths
    )
    salt = np.tile(salttemp[:, np.newaxis], [1, ntimes])
    dsd = xr.Dataset()
    dsd["date_time"] = ("date_time", times, {"axis": "T"})
    dsd["depths"] = ("depths", depths, {"axis": "Z"})
    dsd["lon"] = ("lon", [lon], {"standard_name": "longitude", "axis": "X"})
    dsd["lat"] = ("lat", [lat], {"standard_name": "latitude", "axis": "Y"})
    dsd["temp"] = (("date_time", "depths"), temp.T)
    dsd["salt"] = (("date_time", "depths"), salt.T)
    dds["timeSeriesProfile"] = dsd

    # HF Radar
    example_area = ds.sel(eta_rho=slice(20, 25), xi_rho=slice(10, 15)).isel(
        ocean_time=0, s_rho=-1
    )
    temp = example_area["temp"].interp(
        eta_rho=[20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24.5, 25],
        xi_rho=[10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14],
    )
    salt = example_area["salt"].interp(
        eta_rho=[20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24.5, 25],
        xi_rho=[10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14],
    )
    lons = example_area["lon_rho"].interp(
        eta_rho=[20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24.5, 25],
        xi_rho=[10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14],
    )
    lats = example_area["lat_rho"].interp(
        eta_rho=[20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24.5, 25],
        xi_rho=[10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14],
    )
    dsd = xr.Dataset()
    dsd["temp"] = temp
    dsd["salt"] = salt
    dsd["z_rho"] = 0
    dds["grid"] = dsd

    return dds
