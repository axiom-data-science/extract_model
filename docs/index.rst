.. gcm-filters documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:24:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`extract_model`
===============

Use `extract_model` to read select output from model output files by time and/or space. Output will be selected using `xarray` by a combination of interpolation and index selection. Horizontal interpolation is accomplished using `xESMF` and time interpolation is done with `xarray`'s native 1D interpolation. Currently vertical interpolation is only possible using `xarray`'s 1D interpolation too and is not set up to interpolate in 4D as would be required for ROMS output if not simply selecting the surface layer.

To install from conda-forge:

  >>> conda install -c conda-forge ocean-model-skill-assessor

To install from PyPI:

  >>> pip install cf-pandas

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples and demos

   models.md
   unstructured_subsetting.md
   ts_work.md
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Develop docs

   whats_new



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
