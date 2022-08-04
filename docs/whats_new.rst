:mod:`What's New`
----------------------------

v0.8.1 (Unreleased)
===================

* Support for SELFE datasets is now incorporated into `em.sub_grid()` `em.sub_bbox()` and
  `em.filter()`.

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
======================

* `em.sel2d()` now uses `xoak` to find the nearest neighbor grid point on ND grids. Due to this change, `em.argsel2d()` doesn't exist anymore. Note that vertical functionality that was previously in `em.sel2d()` is now in `em.selZ()`.
* Provide more options in `em.filter()` for keeping different coordinates in a `Dataset`.
* Improvement to unit test setup.
* `em.preprocess()` will implicitly assign horizontal coordinates longitude and latitude for POM
  datasets, even if the data do not specify `coordinates` attributes explicitly.
