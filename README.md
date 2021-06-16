extract_model
==============================
[![Build Status](https://img.shields.io/github/workflow/status/axiom-data-science/extract_model/Tests?logo=github&style=for-the-badge)](https://github.com/axiom-data-science/extract_model/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/extract_model.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/extract_model)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/extract_model/latest.svg?style=for-the-badge)](https://extract_model.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/axiom-data-science/extract_model/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/extract_model/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/extract_model.svg?style=for-the-badge)](https://anaconda.org/conda-forge/extract_model)


Facilitates read-in and extraction for ocean model output.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>

## Installation

### From conda-forge

```
conda install -c conda-forge extract_model
```

### With environment

Clone the repo:
``` bash
$ git clone https://github.com/axiom-data-science/extract_model.git
```

In the `extract_model` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
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
