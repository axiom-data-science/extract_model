"""
Utilities to help extract_model work better.
"""

import numpy as np
import xarray as xr


def sub_grid(ds, bbox):
    """Subset Dataset grids.

    Preserves horizontal grid structure, which matters for ROMS.
    Like `sub_bbox`, this function takes in a bounding box to
    limit the horizontal domain size. But this function preserves
    the relative sizes of horizontal grids if more than 1. There
    will be grid values outside of bbox if grid is curvilinear since
    the axes then do not follow lon/lat directly.

    If there is only one horizontal grid, this simply calls `sub_bbox()`.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset to select model output from.
    bbox: list
        The bounding box for subsetting is defined as [min_lon, min_lat, max_lon, max_lat]

    Returns
    -------
    Dataset that preserves original Dataset horizontal grid structure.

    """

    assertion = "Input should be `xarray` Dataset. For DataArray, try `sub_bbox()`."
    assert isinstance(ds, xr.Dataset), assertion

    attrs = ds.attrs

    # return var names for longitude — may be more than 1
    lons = ds.cf[['longitude']]
    lon_names = set(lons.coords) - set(lons.dims)

    # if longitude is also the name of the dimension
    if len(lon_names) == 0:
        lon_names = set(lons.coords)

    # check for ROMS special case
    if 'lon_rho' in lon_names:
        # variables with 'lon_rho', just use first one
        Var = [Var for Var in ds.data_vars if 'lon_rho' in ds[Var].coords][0]
        subsetted = sub_bbox(ds[Var], bbox, drop=True)
        # import pdb; pdb.set_trace()
        # get xi_rho and eta_rho slice values
        xi_rho, eta_rho = subsetted.xi_rho.values, subsetted.eta_rho.values
        # This first part is to keep the dimensions consistent across
        # the grids
        # then know xi_u, eta_v
        sel_dict = {'xi_rho': xi_rho, 'eta_rho': eta_rho}
        if 'xi_u' in ds:
            sel_dict['xi_u'] = xi_rho[:-1]
        if 'eta_v' in ds:
            sel_dict['eta_v'] = eta_rho[:-1]
        if 'eta_u' in ds:
            sel_dict['eta_u'] = eta_rho
        if 'xi_v' in ds:
            sel_dict['xi_v'] = xi_rho
        if 'eta_psi' in ds:
            sel_dict['eta_psi'] = eta_rho[:-1]
        if 'xi_psi' in ds:
            sel_dict['xi_psi'] = xi_rho[:-1]
        # adjust dimensions of full dataset
        ds_new = ds.sel(sel_dict)

    elif len(lon_names) == 1:

        ds_new = sub_bbox(ds, bbox)

    else:
        # raise exception
        print('Not prepared to deal with this situation.')

    ds_new.attrs = attrs

    return ds_new


def sub_bbox(da, bbox, drop=True):
    """Subset DataArray in space.

    Can also be used on a Dataset if there is only one horizontal grid.

    Parameters
    ----------
    da: DataArray
        Property to select model output from.
    bbox: list
        The bounding box for subsetting is defined as [min_lon, min_lat, max_lon, max_lat]
    drop: bool, optional
        This is passed onto xarray's `da.where()` function. If True, coordinates outside bbox
        are dropped from the DataArray, otherwise they are kept but masked/nan'ed.

    Notes
    -----
    Not dealing with MOM6 output currently.
    """

    # this condition defines the region of interest
    box = (
        (bbox[0] < da.cf["longitude"])
        & (da.cf["longitude"] < bbox[2])
        & (bbox[1] < da.cf["latitude"])
        & (da.cf["latitude"] < bbox[3])
    ).compute()

    da_smaller = da.where(box, drop=drop)

    return da_smaller


def order(da):
    """Reorder var to typical dimensional ordering.

    Parameters
    ------
    da: DataArray
        Variable to operate on.

    Returns
    -------
    DataArray with dimensional order ['T', 'Z', 'Y', 'X'], or whatever subset of
    dimensions are present in da.

    Notes
    -----
    Does not consider previously-selected dimensions that are kept on as coordinates but
    cannot be transposed anymore. This is accomplished with `.reset_coords(drop=True)`.

    Example usage
    -------------
    >>> em.order(da)
    """

    return da.cf.transpose(
        *[
            dim
            for dim in ["T", "Z", "Y", "X"]
            if dim in da.reset_coords(drop=True).cf.axes
        ]
    )


def preprocess_roms(ds):
    """Preprocess ROMS model output for use with cf-xarray.

    Also fixes any other known issues with model output.

    Parameters
    ----------
    ds: xarray Dataset

    Returns
    -------
    Same Dataset but with some metadata added and/or altered.
    """

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

    # calculate vertical coord
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

    # fix attrs
    for zname in ["z_rho", "z_w"]:  # name_dict.values():
        if zname in ds:
            ds[
                zname
            ].attrs = {}  # coord inherits from one of the vars going into calculation
            ds[zname].attrs["positive"] = "up"
            ds[zname].attrs["units"] = "m"
            ds[zname] = order(ds[zname])

    # easier to remove "coordinates" attribute from any variables than add it to all
    for var in ds.data_vars:
        if "coordinates" in ds[var].encoding:
            del ds[var].encoding["coordinates"]

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

    return ds


def preprocess_fvcom(ds):
    """Preprocess FVCOM model output."""

    raise NotImplementedError


def preprocess_selfe(ds):
    """Preprocess SELFE model output."""

    raise NotImplementedError


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


def preprocess_pom(ds):
    """Preprocess POM model output for use with cf-xarray.

    Also fixes any other known issues with model output.

    Parameters
    ----------
    ds: xarray Dataset

    Returns
    -------
    Same Dataset but with some metadata added and/or altered.
    """

    # need to also make this a coordinate to add attributes
    ds["nx"] = ("nx", np.arange(ds.sizes["nx"]), {"axis": "X"})
    ds["ny"] = ("ny", np.arange(ds.sizes["ny"]), {"axis": "Y"})

    ds.cf.decode_vertical_coords(outnames={"sigma": "z"})

    # fix attrs
    for zname in ["z"]:  # name_dict.values():
        if zname in ds:
            ds[
                zname
            ].attrs = {}  # coord inherits from one of the vars going into calculation
            ds[zname].attrs["positive"] = "up"
            ds[zname].attrs["units"] = "m"
            ds[zname] = order(ds[zname])

    return ds


def preprocess_rtofs(ds):
    """Preprocess RTOFS model output."""

    raise NotImplementedError


def preprocess(ds):
    """A preprocess function for reading in with xarray.

    This tries to address known model shortcomings in a generic way so that
    `cf-xarray` will work generally, including decoding vertical coordinates.
    """

    if "ROMS" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_roms(ds)
    elif "FVCOM" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_fvcom(ds)
    elif "SELFE" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_selfe(ds)
    elif "HYCOM" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_hycom(ds)
    elif "POM" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_pom(ds)
    elif "RTOFS" in "".join([str(val) for val in ds.attrs.values()]):
        return preprocess_rtofs(ds)

    return ds
