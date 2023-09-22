"""Preprocessing-related functions for model output."""


from typing import Optional

import numpy as np
import xarray as xr

from extract_model.model_type import ModelType

from .utils import guess_model_type, order


def preprocess_roms(
    ds,
    grid=None,
):
    """Preprocess ROMS model output for use with cf-xarray.

    Also fixes any other known issues with model output.

    Parameters
    ----------
    ds: xarray Dataset

    grid: optional
        Input xgcm grid to have logic run to make Dataset lazily aware of 4D z_rho and z_w coords on u, v, and psi grids.

    Returns
    -------
    Same Dataset but with some metadata added and/or altered.
    """

    rename = {}
    if "eta_u" in ds.dims:
        rename["eta_u"] = "eta_rho"
    if "xi_v" in ds.dims:
        rename["xi_v"] = "xi_rho"
    if "xi_psi" in ds.dims:
        rename["xi_psi"] = "xi_u"
    if "eta_psi" in ds.dims:
        rename["eta_psi"] = "eta_v"
    ds = ds.rename(rename)

    # add axes attributes for dimensions
    dims = [dim for dim in ds.dims if dim.startswith("s_")]
    for dim in dims:
        ds[dim].attrs["axis"] = "Z"

    if "ocean_time" in ds.keys():
        ds.ocean_time.attrs["axis"] = "T"
        ds.ocean_time.attrs["standard_name"] = "time"
    elif "time" in ds.keys():
        ds.time.attrs["axis"] = "T"
        ds.time.attrs["standard_name"] = "time"

    dims = [dim for dim in ds.dims if dim.startswith("xi_")]
    # need to also make this a coordinate to add attributes
    for dim in dims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {"axis": "X"})

    dims = [dim for dim in ds.dims if dim.startswith("eta_")]
    for dim in dims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {"axis": "Y"})

    # add attributes for lon/lat
    lon_attrs = {
        "standard_name": "longitude",
        "units": "degree_east",
        "field": "longitude",
    }
    lat_attrs = {
        "standard_name": "latitude",
        "units": "degree_north",
        "field": "latitude",
    }
    coords = [coord for coord in ds.coords if coord.startswith("lon_")]
    for coord in coords:
        ds[coord].attrs = lon_attrs
    coords = [coord for coord in ds.coords if coord.startswith("lat_")]
    for coord in coords:
        ds[coord].attrs = lat_attrs

    # Fix standard_name for s_rho/s_w
    if "Vtransform" in ds.data_vars and "s_rho" in ds.coords:
        cond1 = (
            ds["Vtransform"] == 1
            and ds["s_rho"].attrs["standard_name"] == "ocean_s_coordinate"
        )
        cond2 = (
            ds["Vtransform"] == 2
            and ds["s_rho"].attrs["standard_name"] == "ocean_s_coordinate"
        )
        if cond1:
            ds["s_rho"].attrs["standard_name"] = "ocean_s_coordinate_g1"
        elif cond2:
            ds["s_rho"].attrs["standard_name"] = "ocean_s_coordinate_g2"

        cond1 = (
            ds["Vtransform"] == 1
            and ds["s_w"].attrs["standard_name"] == "ocean_s_coordinate"
        )
        cond2 = (
            ds["Vtransform"] == 2
            and ds["s_w"].attrs["standard_name"] == "ocean_s_coordinate"
        )
        if cond1:
            ds["s_w"].attrs["standard_name"] = "ocean_s_coordinate_g1"
        elif cond2:
            ds["s_w"].attrs["standard_name"] = "ocean_s_coordinate_g2"

    # create vertical coordinates z_rho and z_w
    name_dict = {}
    if "s_rho" in ds.dims:
        name_dict["s_rho"] = "z_rho"
        if "positive" in ds.s_rho.attrs:
            ds.s_rho.attrs.pop("positive")
    if "s_w" in ds.dims:
        name_dict["s_w"] = "z_w"
        if "positive" in ds.s_w.attrs:
            ds.s_w.attrs.pop("positive")
    ds.cf.decode_vertical_coords(outnames=name_dict)

    # expand Z coordinates to u and v grids
    if grid is not None:
        # necessary for interpolating u and v to depths
        # ds.coords["z_w"] = order(ds["z_w"])
        # ds.coords["z_w_u"] = grid.interp(ds.z_w.chunk({ds.z_w.cf["X"].name: -1}), "X")
        # ds.coords["z_w_u"].attrs = {
        #     "long_name": "depth of U-points on vertical W grid",
        #     "time": "ocean_time",
        #     "field": "z_w_u, scalar, series",
        #     "units": "m",
        # }
        # ds.coords["z_w_v"] = grid.interp(ds.z_w.chunk({ds.z_w.cf["Y"].name: -1}), "Y")
        # ds.coords["z_w_v"].attrs = {
        #     "long_name": "depth of V-points on vertical W grid",
        #     "time": "ocean_time",
        #     "field": "z_w_v, scalar, series",
        #     "units": "m",
        # }
        # ds.coords["z_w_psi"] = grid.interp(ds.z_w_u.chunk({ds.z_w_u.cf["Y"].name: -1}), "Y")
        # ds.coords["z_w_psi"].attrs = {
        #     "long_name": "depth of PSI-points on vertical W grid",
        #     "time": "ocean_time",
        #     "field": "z_w_psi, scalar, series",
        #     "units": "m",
        # }

        ds.coords["z_rho"] = order(ds["z_rho"])
        ds.coords["z_rho_u"] = grid.interp(
            ds.z_rho.chunk({ds.z_rho.cf["X"].name: -1}), "X"
        )
        ds.coords["z_rho_u"].attrs = {
            "long_name": "depth of U-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_u, scalar, series",
            "units": "m",
        }

        ds.coords["z_rho_v"] = grid.interp(
            ds.z_rho.chunk({ds.z_rho.cf["Y"].name: -1}), "Y"
        )
        ds.coords["z_rho_v"].attrs = {
            "long_name": "depth of V-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_v, scalar, series",
            "units": "m",
        }

        ds.coords["z_rho_psi"] = grid.interp(
            ds.z_rho_u.chunk({ds.z_rho_u.cf["Y"].name: -1}), "Y"
        )
        ds.coords["z_rho_psi"].attrs = {
            "long_name": "depth of PSI-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_psi, scalar, series",
            "units": "m",
        }

        # will use this to update coordinate encoding
        name_dict.update(
            {"filler1": "z_rho_u", "filler2": "z_rho_v", "filler3": "z_rho_psi"}
        )  # , "None": "z_w_u", "None": "z_w_v", "None": "z_w_psi"})

    # fix attrs
    # for zname in ["z_rho", "z_w"]:
    for zname in [var for var in ds.coords if "z_rho" in var or "z_w" in var]:
        if zname in ds:
            ds[
                zname
            ].attrs = {}  # coord inherits from one of the vars going into calculation
            ds[zname].attrs["positive"] = "up"
            ds[zname].attrs["units"] = "m"
            ds[zname] = order(ds[zname])

    # replace s_rho with z_rho, etc, to make z_rho the vertical coord
    for var in ds.data_vars:
        if ds[var].ndim >= 3:
            if "coordinates" in ds[var].encoding:
                coords = ds[var].encoding["coordinates"]
                # update s's to z's
                for sname, zname in name_dict.items():
                    if sname in coords:  # replace if present
                        coords = coords.replace(sname, zname)
                    else:  # still add z_rho or z_w
                        if zname in ds[var].coords and ds[zname].shape == ds[var].shape:
                            coords += f" {zname}"
                # could have "x_rho" instead of "lon_rho", etc
                if "x_" in coords:
                    xcoord = [element for element in coords.split() if "x_" in element][
                        0
                    ]
                    coords = coords.replace(xcoord, xcoord.replace("x", "lon"))
                # could have "x_rho" instead of "lon_rho", etc
                if "y_" in coords:
                    ycoord = [element for element in coords.split() if "y_" in element][
                        0
                    ]
                    coords = coords.replace(ycoord, ycoord.replace("y", "lat"))
                ds[var].encoding["coordinates"] = coords
            # same but coordinates not inside encoding. Do same processing
            # but also move coordinates from attrs to encoding.
            elif "coordinates" in ds[var].attrs:
                coords_here = ds[var].attrs["coordinates"]
                # update s's to z's
                for sname, zname in name_dict.items():
                    if sname in coords_here:  # replace if present
                        coords_here = coords_here.replace(sname, zname)
                    else:  # still add z_rho or z_w
                        if zname in ds[var].coords and ds[zname].shape == ds[var].shape:
                            coords_here += f" {zname}"
                # could have "x_rho" instead of "lon_rho", etc
                if "x_" in coords_here:
                    xcoord = [
                        element for element in coords_here.split() if "x_" in element
                    ][0]
                    coords_here = coords_here.replace(
                        xcoord, xcoord.replace("x", "lon")
                    )
                # could have "x_rho" instead of "lon_rho", etc
                if "y_" in coords_here:
                    ycoord = [
                        element for element in coords_here.split() if "y_" in element
                    ][0]
                    coords_here = coords_here.replace(
                        ycoord, ycoord.replace("y", "lat")
                    )
                # move coords to encoding and delete from attrs
                ds[var].encoding["coordinates"] = coords_here
                del ds[var].attrs["coordinates"]

    # # easier to remove "coordinates" attribute from any variables than add it to all
    # for var in ds.data_vars:
    #     if "coordinates" in ds[var].encoding:
    #         del ds[var].encoding["coordinates"]

    #     # add attribute "coordinates" to all variables with at least 2 dimensions
    #     # and the dimensions have to be the regular types (time, Z, Y, X)
    #     for var in ds.data_vars:
    #         if ds[var].ndim >= 2 and (len(set(ds[var].dims) - set([ds[var].cf[axes].name for axes in ds[var].cf.axes])) == 0):
    #             coords = ['time', 'vertical', 'latitude', 'longitude']
    #             var_names = [ds[var].cf[coord].name for coord in coords if coord in ds[var].cf.coords.keys()]
    #             coord_str = " ".join(var_names)
    #             ds[var].attrs["coordinates"] = coord_str

    # Add standard_names for typical ROMS variables
    # should this not overwrite standard name if it already exists?
    var_map = {
        "zeta": "sea_surface_elevation",
        "salt": "sea_water_practical_salinity",
        "temp": "sea_water_temperature",
    }
    for var_name, standard_name in var_map.items():
        if var_name in ds.data_vars and "standard_name" not in ds[var_name].attrs:
            ds[var_name].attrs["standard_name"] = standard_name

    # Fix calendar if wrong
    attrs = ds[ds.cf["T"].name].attrs
    if ("calendar" in attrs) and (attrs["calendar"] == "gregorian_proleptic"):
        attrs["calendar"] = "proleptic_gregorian"
        ds[ds.cf["T"].name].attrs = attrs

    if "s_rho" in ds.dims:
        if "positive" in ds.s_rho.attrs:
            ds.s_rho.attrs.pop("positive")
    if "s_w" in ds.dims:
        if "positive" in ds.s_w.attrs:
            ds.s_w.attrs.pop("positive")

    return ds


def preprocess_roms_grid(ds):
    # use xgcm
    from xgcm import Grid

    coords = {
        "X": {"center": "xi_rho", "inner": "xi_u"},
        "Y": {"center": "eta_rho", "inner": "eta_v"},
        "Z": {"center": "s_rho", "outer": "s_w"},
    }
    grid = Grid(ds, coords=coords, periodic=False)
    return grid


def preprocess_fvcom(ds):
    """Preprocess FVCOM model output."""
    return ds


def preprocess_selfe(ds):
    """Preprocess SELFE model output."""
    return ds


def preprocess_hycom(ds):
    """Preprocess HYCOM model output for use with cf-xarray.

    Also fixes any other known issues with model output.

    Parameters
    ----------
    ds: xarray Dataset

    Returns
    -------
    Same Dataset but with some metadata added and/or altered.
    """

    if "time" in ds:
        ds["time"].attrs["axis"] = "T"

    return ds


def preprocess_pom(ds, interp_vertical: bool = True):
    """Preprocess POM model output for use with cf-xarray.

    Also fixes any other known issues with model output.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset containing data described from POM output.

    Returns
    -------
    xr.Dataset
        Same Dataset but with some metadata added and/or altered.
    """
    # The longitude and latitude variables are not recognized as valid coordinates
    if "longitude" not in ds.cf.coords:
        if "longitude" not in ds.cf.standard_names:
            raise ValueError("No variable describing longitude is available.")

        if "latitude" not in ds.cf.standard_names:
            raise ValueError("No variable describing latitude is available.")

        ds = ds.cf.set_coords(["latitude", "longitude"])

    # need to also make this a coordinate to add attributes
    ds["nx"] = ("nx", np.arange(ds.sizes["nx"]), {"axis": "X"})
    ds["ny"] = ("ny", np.arange(ds.sizes["ny"]), {"axis": "Y"})

    # need to add coordinates to each data variable too
    for var in ds.data_vars:
        if ds[var].ndim == 3:
            ds[var].encoding["coordinates"] = "time lat lon"
        elif ds[var].ndim == 4:
            ds[var].encoding["coordinates"] = "time depth lat lon"

    if interp_vertical:
        ds.cf.decode_vertical_coords(outnames={"sigma": "z"})

        # fix attrs
        for zname in ["z"]:  # name_dict.values():
            if zname in ds:
                ds[
                    zname
                ].attrs = (
                    {}
                )  # coord inherits from one of the vars going into calculation
                ds[zname].attrs["positive"] = "up"
                ds[zname].attrs["units"] = "m"
                ds[zname] = order(ds[zname])

    # keep sigma from showing up as "vertical" in cf-xarray
    for sname in ["sigma"]:  # name_dict.values():
        if sname in ds:
            del ds[sname].attrs["positive"]

    return ds


def preprocess_rtofs(ds):
    """Preprocess RTOFS model output."""

    raise NotImplementedError


def preprocess(ds, model_type=None, kwargs=None):
    """A preprocess function for reading in with xarray.

    This tries to address known model shortcomings in a generic way so that
    `cf-xarray` will work generally, including decoding vertical coordinates.
    """

    kwargs = kwargs or {}

    # This is an internal attribute used by netCDF which xarray doesn't know or care about, but can
    # be returned from THREDDS.
    if "_NCProperties" in ds.attrs:
        del ds.attrs["_NCProperties"]

    # Preprocess for all models: if cf-xarray has not identifed axes Z but has identified coordinate vertical
    # and the vertical coordinate is 1D, add `axis="Z"` to its attributes so it will also be recognized as
    # the Z axes.
    if "vertical" in ds.cf.coordinates and "Z" not in ds.cf.axes:
        if ds.cf["vertical"].ndim == 1 and len(ds.cf.coordinates["vertical"]) == 1:
            key = ds.cf.coordinates["vertical"][0]
            ds[key].attrs["axis"] = "Z"

    preprocess_map = {
        "ROMS": preprocess_roms,
        "FVCOM": preprocess_fvcom,
        "SELFE": preprocess_selfe,
        "HYCOM": preprocess_hycom,
        "POM": preprocess_pom,
        "RTOFS": preprocess_rtofs,
    }

    if model_type is None:
        model_type = guess_model_type(ds)

    if model_type in preprocess_map:
        return preprocess_map[model_type](ds, **kwargs)

    return ds
