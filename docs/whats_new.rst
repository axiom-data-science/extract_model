:mod:`What's New`
----------------------------

v1.0.0 (July 22, 2022)
======================

* `em.sel2d()` now uses `xoak` to find the nearest neighbor grid point on ND grids. Due to this change, `em.argsel2d()` doesn't exist anymore. Note that vertical functionality that was previously in `em.sel2d()` is now in `em.selZ()`.
* Provide more options in `em.filter()` for keeping different coordinates in a `Dataset`.
* Improvement to unit test setup.
