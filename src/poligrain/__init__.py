"""Copyright (c) 2023 Christian Chwala. All rights reserved.

poligrain: Effortlessly plot and compare (rainfall) sensor data
           with point, line and grid geometry.
"""


from __future__ import annotations

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray
import netCDF4  # noqa: F401

__version__ = "0.0.0"

from . import plot_map, spatial

__all__ = ["__version__", "plot_map", "spatial"]
