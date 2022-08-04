:mod:`What's New`
----------------------------
v0.8.1
======

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
