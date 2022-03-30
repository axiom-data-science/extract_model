"""
Utilities to help extract_model work better.
"""

import numpy as np


def subset(da, bbox, drop=True):
    """Subset DataArray 
    
    bbox = [min_lon, min_lat, max_lon, max_lat]
    UPDATE
    """

    # this condition defines the region of interest
    box = ((bbox[0] < da.cf['longitude']) & (da.cf['longitude'] < bbox[2]) \
           & (bbox[1] < da.cf['latitude']) & (da.cf['latitude'] < bbox[3])).compute()

    da_smaller = da.where(box, drop=drop)
    
    return da_smaller


def order(var):
    """Reorder var to typical dimensional ordering.
    Inputs
    ------
    var: DataArray
        Variable to operate on.
    Returns
    -------
    DataArray with dimensional order ['T', 'Z', 'Y', 'X'], or whatever subset of
    dimensions are present in var.
    Notes
    -----
    Do not consider previously-selected dimensions that are kept on as coordinates but
    cannot be transposed anymore. This is accomplished with `.reset_coords(drop=True)`.
    Example usage
    -------------
    >>> xroms.order(var)
    """

    return var.cf.transpose(
        *[
            dim
            for dim in ["T", "Z", "Y", "X"]
            if dim in var.reset_coords(drop=True).cf.axes
        ]
    )


def preprocess_roms(ds):
    """Preprocess ROMS model output."""
    
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
    if 'Vtransform' in ds.data_vars and 's_rho' in ds.coords:
        cond1 = ds['Vtransform'] == 1 and ds['s_rho'].attrs['standard_name'] == 'ocean_s_coordinate'
        cond2 = ds['Vtransform'] == 2 and ds['s_rho'].attrs['standard_name'] == 'ocean_s_coordinate'
        if cond1:
            ds['s_rho'].attrs['standard_name'] = 'ocean_s_coordinate_g1'
        elif cond2:
            ds['s_rho'].attrs['standard_name'] = 'ocean_s_coordinate_g2'

        cond1 = ds['Vtransform'] == 1 and ds['s_w'].attrs['standard_name'] == 'ocean_s_coordinate'
        cond2 = ds['Vtransform'] == 2 and ds['s_w'].attrs['standard_name'] == 'ocean_s_coordinate'
        if cond1:
            ds['s_w'].attrs['standard_name'] = 'ocean_s_coordinate_g1'
        elif cond2:
            ds['s_w'].attrs['standard_name'] = 'ocean_s_coordinate_g2'
            
    # calculate vertical coord
    name_dict = {}
    if 's_rho' in ds.dims:
        name_dict['s_rho'] = 'z_rho'
        if "positive" in ds.s_rho.attrs:
            ds.s_rho.attrs.pop("positive")
    if 's_w' in ds.dims:
        name_dict['s_w'] = 'z_w'
        if "positive" in ds.s_w.attrs:
            ds.s_w.attrs.pop("positive")
    ds.cf.decode_vertical_coords(outnames=name_dict)

    # fix attrs
    for zname in ['z_rho', 'z_w']:# name_dict.values():
        if zname in ds:
            ds[zname].attrs = {}  # coord inherits from one of the vars going into calculation
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
    var_map = {'zeta': 'sea_surface_elevation',
               'salt': 'sea_water_practical_salinity',
               'temp': 'sea_water_temperature'}
    for var_name, standard_name in var_map.items():
        if var_name in ds.data_vars and 'standard_name' not in ds[var_name].attrs:
            ds[var_name].attrs['standard_name'] = standard_name
    
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
    """Preprocess HYCOM model output."""

    return ds


def preprocess_pom(ds):
    """Preprocess POM model output."""

    # need to also make this a coordinate to add attributes
    ds['nx'] = ('nx', np.arange(ds.sizes['nx']), {"axis": "X"})
    ds['ny'] = ('ny', np.arange(ds.sizes['ny']), {"axis": "Y"})

    ds.cf.decode_vertical_coords(outnames={'sigma': 'z'})

    # fix attrs
    for zname in ['z']:# name_dict.values():
        if zname in ds:
            ds[zname].attrs = {}  # coord inherits from one of the vars going into calculation
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
    
    if 'ROMS' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_roms(ds)
    elif 'FVCOM' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_fvcom(ds)
    elif 'SELFE' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_selfe(ds)
    elif 'HYCOM' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_hycom(ds)
    elif 'POM' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_pom(ds)
    elif 'RTOFS' in ''.join([str(val) for val in ds.attrs.values()]):
        return preprocess_rtofs(ds)
    
            
#     try:
#         ds.cf.decode_vertical_coords()
#     except Exception:
#         print('Could not decode vertical coordinates.')
#         pass
    
#     ds = ds.cf.guess_coord_axis()

    return ds
