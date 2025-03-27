# CHANGELOG

## unreleased

### New features

- [PR95](https:github.com/OpenSenseAction/poligrain/pull/95) allow x-y
  coordinates and use it as default in GridAtPoints and GridAtLines (by
  [@cchwala](https://github.com/cchwala))
- [PR94](https:github.com/OpenSenseAction/poligrain/pull/94) add option to pass
  data without time dimension to `GridAtPoints` and `GridAtLines` (by
  [@cchwala](https://github.com/cchwala))
- [PR90](https://github.com/OpenSenseAction/poligrain/pull/90) add function to
  download 8-day example datasets for OpenRainER (by
  [@cchwala](https://github.com/cchwala))
- [PR89](https://github.com/OpenSenseAction/poligrain/pull/89) add function to
  download 8-day example datasets for OpenMRG with processed CML data (by
  [@cchwala](https://github.com/cchwala))
- [PR 40](https://github.com/OpenSenseAction/poligrain/pull/40) Add functions
  for plotting CML metadata (by [@bwalraven](https://github.com/bwalraven))

### Bug fixes

### Maintenance

### Breaking changes

## v0.2.2

Add back support for Python 3.9.

## v0.2.1

This is just a quick release to start tracking via zenodo to get a DOI.

## v0.2.0

### New features

- [PR75](https://github.com/OpenSenseAction/poligrain/pull/75) Add function to
  easily plot grid, line and point data (by
  [@cchwala](https://github.com/cchwala))
- [PR76](https://github.com/OpenSenseAction/poligrain/pull/76) add function to
  download small example datasets from newly created data repo at
  https://github.com/cchwala/opensense_example_data/tree/main (by
  [@cchwala](https://github.com/cchwala))

### Bug fixes

- [PR74](https://github.com/OpenSenseAction/poligrain/pull/74) fix xarray
  scatter plot CI error in notebook (by [@cchwala](https://github.com/cchwala))

### Maintenance

- [PR74](https://github.com/OpenSenseAction/poligrain/pull/74) remove version
  capping (by [@cchwala](https://github.com/cchwala))

### Breaking changes

## v0.1.1

This is just a quick update to make a version available with less strict
dependencies, which were relaxed in
[PR 66](https://github.com/OpenSenseAction/poligrain/pull/66).

Note that support for Python 3.9 was dropped and support for versions newer than
3.11 was added.

Testing is now done for 3.10, 3.11 and 3.12.

## v0.1.0

With all the new features from the last versions and because we seem to be on
the right track regarding structure of functions and modules, we now switch to
v0.1.0.

### New features

- [PR 40](https://github.com/OpenSenseAction/poligrain/pull/40) Add functions
  for plotting CML metadata (by [@bwalraven](https://github.com/bwalraven))

### Bug fixes

### Maintenance

### Breaking changes

## v0.0.5

### New features

- [PR 43](https://github.com/OpenSenseAction/poligrain/pull/43) Add code for
  finding closest points to lines (by [@eoydvin](https://github.com/eoydvin))
- [PR 54](https://github.com/OpenSenseAction/poligrain/pull/54) Add example code
  and notebook to showcase the grid intersection between radar data and CMLs (by
  [@maxmargraf](https://github.com/maxmargraf))
- [PR 55](https://github.com/OpenSenseAction/poligrain/pull/55) Add recipe for
  converting CSV data to xarray.Dataset (by
  [@JochenSeidel](https://github.com/jochenseidel))
- [PR 41](https://github.com/OpenSenseAction/poligrain/pull/41) Add simple API
  to get GridAtLines and GridAtPoints (by
  [@cchwala](https://github.com/cchwala))

### Bug fixes

### Maintenance

### Breaking changes

## v0.0.4

### New features

- [PR 49](https://github.com/OpenSenseAction/poligrain/pull/49) Add
  point-to_point nearest neighbor lookup (by
  [@cchwala](https://github.com/cchwala))
- [PR 24](https://github.com/OpenSenseAction/poligrain/pull/24) Allow to color
  CML paths when plotting on map using a cmap (by
  [@cchwala](https://github.com/cchwala))
- [PR 29](https://github.com/OpenSenseAction/poligrain/pull/29) Add `xarray`
  Accessor for plotting CML paths (by [@cchwala](https://github.com/cchwala))
- [PR 19](https://github.com/OpenSenseAction/poligrain/pull/19) Add code for
  grid intersection calculation (by
  [@maxmargraf](https://github.com/maxmargraf))

### Bug fixes

### Maintenance

- [PR 23](https://github.com/OpenSenseAction/poligrain/pull/23) Make syntax
  highliting work on readthedocs (by [@cchwala](https://github.com/cchwala))
- [PR 31](https://github.com/OpenSenseAction/poligrain/pull/31) Fix pandoc
  dependencies for local build of documentation (by
  [@cchwala](https://github.com/cchwala))
- [PR 33](https://github.com/OpenSenseAction/poligrain/pull/33) Add numpy
  docstring linting rules (by [@cchwala](https://github.com/cchwala))
- [PR 36](https://github.com/OpenSenseAction/poligrain/pull/36) Add API to docs
  (by [@cchwala](https://github.com/cchwala))
- [PR 42](https://github.com/OpenSenseAction/poligrain/pull/42) Add back support
  for Python 3.9 (by [@cchwala](https://github.com/cchwala))

### Breaking changes

## v0.0.3

### New features

- [PR 2](https://github.com/OpenSenseAction/poligrain/pull/2) Add first version
  of plotting functions for CML paths (by
  [@cchwala](https://github.com/cchwala))
- [PR 16](https://github.com/OpenSenseAction/poligrain/pull/16) Add calculation
  of point-to-point distances and added example notebook (by
  [@cchwala](https://github.com/cchwala))

### Bug fixes

### Maintenance

- [PR 18](https://github.com/OpenSenseAction/poligrain/pull/18) Disable mypy
  because it does not work as expected for now due to problems with envs in
  pre-commit and CI (by [@cchwala](https://github.com/cchwala))
- [PR 7](https://github.com/OpenSenseAction/poligrain/pull/7) Add testing and
  linting of notebooks (by [@cchwala](https://github.com/cchwala))

### Breaking changes

## v0.0.2

not documented...
