#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backend which explicitly supports data defining a triangular mesh."""
from xarray import conventions
from xarray.backends.common import _normalize_path
from xarray.backends.netCDF4_ import NetCDF4BackendEntrypoint, NetCDF4DataStore
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataset import Dataset
from xarray.core.utils import close_on_error


class TriangularMeshNetCDF4StoreEntrypoint(StoreBackendEntrypoint):
    """A custom StoreBackendEntrypoint that enables clients to rename variables.

    The purpose of this class is to enable clients to specify a mapping to rename variables after
    the store has loaded the xarray objects but before they are assembled into a Dataset. This
    enables loading of complex structures like triangular meshes where variables that appear as
    coordinate variables but have multiple dimensions could be forgiven by renaming them.
    """

    def open_dataset(
        self,
        store,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        preload_varmap=None,
    ):
        """Return an xr.Dataset after renaming variables described in preload_varmap."""
        vars, attrs = store.load()
        encoding = store.get_encoding()

        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        if preload_varmap is not None:
            for var_in, var_out in preload_varmap.items():
                vars[var_out] = vars[var_in]
                del vars[var_in]

        ds = Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(store.close)
        ds.encoding = encoding

        return ds


class TriangularMeshNetCDF4BackendEntrypoint(NetCDF4BackendEntrypoint):
    """A custom Backend for xarray to support loading unstructured grids with a triangular mesh."""

    def guess_can_open(self, filename_or_obj) -> bool:
        """Return False, only use this backend by explicit invocation."""
        return False

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        mode="r",
        format="NETCDF4",
        clobber=True,
        diskless=False,
        persist=False,
        lock=None,
        autoclose=False,
        preload_varmap=None,
    ):
        """Return an open xr.Dataset object using the TriangularMeshNetCDF4BackendEntrypoint."""

        filename_or_obj = _normalize_path(filename_or_obj)
        store = NetCDF4DataStore.open(
            filename_or_obj,
            mode=mode,
            format=format,
            group=group,
            clobber=clobber,
            diskless=diskless,
            persist=persist,
            lock=lock,
            autoclose=autoclose,
        )

        store_entrypoint = TriangularMeshNetCDF4StoreEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
                preload_varmap=preload_varmap,
            )
        return ds
