# Installation development
#### 1. Clone the repository

To get started clone the `mammos-units` repository via `ssh`:

```bash
git clone git@github.com:MaMMoS-project/mammos-units.git
```
or `https` if you don't have an `ssh` key:

```bash
git clone https://github.com/MaMMoS-project/mammos-units.git
```

The enter into the repository:

```bash
cd mammos-units
```

### Install dependencies

#### Option 1: with pixi (recommended)

- install [pixi](https://pixi.sh)

- run `pixi shell` to create and activate an environment in which `mammos-units` is installed (this will install python as well)

- Alternatively, to fire up the `example.ipynb` notebook, use `pixi run example`.

#### Option 2: Create and activate `conda` environment, and install `mammos-units` via pip

If required install `conda`. Suggestion: use [miniforge](https://github.com/conda-forge/miniforge).

```bash
conda create -n mammosunits python=3.12 pip
conda activate mammosunits
```

Install a local editable version of the code

```bash
pip install -e .
```
