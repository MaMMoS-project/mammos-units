# mammosunits

`mammosunits` is a units package built on top of the powerful Astropy units.
It provides additional functionality for magnetic units.

## Clone and Install 
#### 1. Clone the repository

To get started clone the `mammosunits` repository via `ssh`:

```bash
git clone git@github.com:MaMMoS-project/units.git
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

#### 3. Install the local version of the `units` package

Install a local editable version of the code

```bash
pip install -e .
```
