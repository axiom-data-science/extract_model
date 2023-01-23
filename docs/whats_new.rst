:mod:`What's New`
----------------------------

v1.1.1 (January 23, 2023)
=========================
* Updated the docs so that the notebooks are compiled with `myst-nb`.
* Updated pull_request_template.md.

v1.1.0 (January 19, 2023)
=========================

Two main changes in `sel2d` / `sel2dcf`:

* a mask can be input to limit the lons/lats from the DataArray/Dataset that are used in searching for the nearest point, in case the nearest model point is on land but we still want a valid model point returned.
* incorporating changes from xoak that optional return distance of the model point(s) from the requested point(s).

v1.0.0 (December 9, 2022)
=========================
* Simplified dependencies
* Now available on PyPI!

v0.9.0 (September 26, 2022)
===========================
* An optional subsetting option that enables subsetting directly on the target
  dataset's dimensions. For remote datasets, this ensures that remote requests
  ask for minimal slices. `em.sub_grid(..., naive=True)`.
* Adds `preload` argument for unstructured grid subsetting. Radically improves xarray resolution
  times after subsetting.

v0.8.1 (August 16, 2022)
========================

* Support for SELFE datasets is now incorporated into `em.sub_grid()` `em.sub_bbox()` and
  `em.filter()`.
* Support for numba and numpy implementations of `index_of_sorted()`.

v0.8.0 (August 3, 2022)
=======================

* `extract_model` has a backend that will support reading in FVCOM model output which has previously
  not been possible when using `xarray` without dropping the vertical grid coordinates.
* `em.sub_bbox()` supports subsetting FVCOM model output.
* A new jupyter notebook demonstrating subsetting of FVCOM model output is now available in docs.
* `em.sub_grid()` supports subsetting FVCOM model output.
* `em.filter()` will not discard any unstructured coordinate information in the auxiliary coordinate
  variables.

v0.7 (July 22, 2022)
====================

* `em.sel2d()` now uses `xoak` to find the nearest neighbor grid point on ND grids. Due to this change, `em.argsel2d()` doesn't exist anymore. Note that vertical functionality that was previously in `em.sel2d()` is now in `em.selZ()`.
* Provide more options in `em.filter()` for keeping different coordinates in a `Dataset`.
* Improvement to unit test setup.
* `em.preprocess()` will implicitly assign horizontal coordinates longitude and latitude for POM
  datasets, even if the data do not specify `coordinates` attributes explicitly.
