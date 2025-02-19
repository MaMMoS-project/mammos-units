# mammosunits

`mammosunits` is a units package built on top of the powerful Astropy units.
It provides additional functionality for magnetic units.

## Try it in the cloud
Try `mammosunits` without installing it locally by directly accessing it directly in the cloud
via Binder.
Simply click the badge to get started: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MaMMoS-project/units/latest?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fexample.ipynb)

Sessions are temporary and may time out after a period of inactivity, and any files
created or modified during your session will not be saved.
To avoid losing your work, please remember to download any files you create or edit
before your session ends.

## Clone and Install 
#### 1. Clone the repository

To get started clone the `mammosunits` repository via `ssh`:

```bash
git clone git@github.com:MaMMoS-project/units.git
```
or `https` if you don't have an `ssh` key:

```bash
git clone https://github.com/MaMMoS-project/units.git
```

The enter into the repository:

```bash
cd units
```

### Install dependencies

#### Option 1: with pixi (recommended)

- install [pixi](https://pixi.sh)

- run `pixi shell` to create and activate an environment in which `units` is installed (this will install python as well)

- Alternatively, to fire up the `example.ipynb` notebook, use `pixi run example`.

#### Option 2: Create and activate `conda` environment

If required install `conda`. Suggestion: use [miniforge](https://github.com/conda-forge/miniforge).

```bash
conda create -n mammosunits python=3.12 pip
conda activate mammosunits
```

Install a local editable version of the code

```bash
pip install -e .
```
