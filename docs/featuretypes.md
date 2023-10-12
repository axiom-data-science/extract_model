# Feature types

Feature types are defined by NCEI and provide structure types of data to expect. More information is available [in general](https://www.ncei.noaa.gov/netcdf-templates) and for the current [NCEI NetCDF Templates 2.0](https://www.ncei.noaa.gov/data/oceans/ncei/formats/netcdf/v2.0/index.html). The following information may be useful for thinking about this. In particular, you select `locstream`, `locstreamT`, and `locstreamZ` as a user for `em.select()` and this table can guide how to select.

|                 | timeSeries     | profile        | timeSeriesProfile | trajectory (TODO)                     | trajectoryProfile     | grid (TODO)         |
|---              |---             |---             |---                |---                                    | ---                   | ---                 |
| Definition      | only t changes | only z changes | t and z change    | t, y, and x change                    | t, z, y, and x change | t changes, y/x grid |
| Data types      | mooring, buoy  | CTD profile    | moored ADCP       | flow through, 2D drifter | glider, transect of CTD profiles, towed ADCP, 3D drifter   | satellite, HF Radar |
| Model extraction | time series at surface or depth |  | vertical cross section | 2D drifters | 3D drifters | regridding, x/y slice in depth |
| X/Y are pairs ("locstream") or grid | either locstream or grid | either locstream or grid | either locstream or grid | locstream | locstream | grid |
| Which dimensions are independent from X/Y choice? |
| T | Independent | Independent | Independent | locstreamT | locstreamT | Independent |
| Z | Independent | Independent | Independent | Independent | locstreamZ | Independent |
