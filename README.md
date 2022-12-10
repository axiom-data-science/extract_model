extract_model
==============================
[![Build Status](https://img.shields.io/github/workflow/status/axiom-data-science/extract_model/Tests?logo=github&style=for-the-badge)](https://github.com/axiom-data-science/extract_model/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/extract_model.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/extract_model)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/extract_model/latest.svg?style=for-the-badge)](https://extract_model.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/axiom-data-science/extract_model/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/extract_model/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/extract_model.svg?style=for-the-badge)](https://anaconda.org/conda-forge/extract_model)
[![Python Package Index](https://img.shields.io/pypi/v/extract_model.svg?style=for-the-badge)](https://pypi.org/project/extract_model)

Facilitates read-in and extraction for ocean model output. Most functions work on both structured and unstructured model output. Unstructured functionality has been tested with FVCOM and SELFE output.

In particular this package can:
- interpolate a model time series to a longitude, latitude location on a 2D grid, while bringing along the calculated z coordinates, with `select()`
 - saves the weights of the interpolation to save time in the accessor if used, or allows user to input
 - uses [`xESMF`](https://pangeo-xesmf.readthedocs.io/en/latest/index.html) for fast interpolation that respects longitude/latitude grids
- find the nearest grid point to a longitude, latitude location on a horizontal grid (structured or unstructured) in `sel2d()` using [`xoak`](https://xoak.readthedocs.io/en/latest/index.html)
 - `xoak` saves the calculated index so that subsequent searches are faster
- select a sub-region of a structured or unstructured grid in two ways with `sub_bbox()` and `sub_grid()`
- has an `xarray` accessor for convenient access to methods
- uses `cf-xarray` to understand `xarray` Dataset metadata and allow for generic axis and coordinate names as well as calculate vertical coordinates
- can preprocess a variety of model output types (including ROMS, HYCOM, POM, and FVCOM) to improve metadata and ease of use

> :warning: **If you are using Windows**: Horizontal interpolation currently will not work in `extract_model` until `xESMF` is installable on Windows. Other functions will work.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>

## Installation

### From conda-forge

This will install for all operating systems:
``` bash
conda install -c conda-forge extract_model
```
but it includes only minimal requirements. If you want to install packages to run the example docs notebooks and to make unstructured grid subsetting faster, you can additionally install for Windows:

``` bash
$ conda install --file conda-requirements-opt-win.txt
```

For running the example docs notebooks, horizontal interpolation (with `xESMF`), and to make horizontal subsetting faster, install additional packages for Mac and Linux:

``` bash
$ conda install --file conda-requirements-opt.txt
```

### From PyPI

``` bash
pip install extract_model
```

### With environment

Clone the repo:
``` bash
$ git clone https://github.com/axiom-data-science/extract_model.git
```

In the `extract_model` directory, install conda environment (for Mac or Linux):
``` bash
$ conda env create -f environment.yml
```
or for Windows:
``` bash
$ conda env create -f environment-win.yml
```

### Local

For local package install, after cloning the repository, in the `extract_model` directory:
``` bash
$ pip install -e .
```

### Development

To also develop this package, install additional packages with:
``` bash
$ conda install --file requirements-dev.txt
```

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```
