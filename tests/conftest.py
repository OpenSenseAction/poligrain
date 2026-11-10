"""Shared pytest fixtures and configuration for the test suite."""

import os

# Use a non-interactive backend so that tests do not require a display or
# Tkinter/Tcl, which is not reliably available on CI runners (e.g. Windows).
#
# We set the MPLBACKEND environment variable instead of calling
# matplotlib.use("Agg") here, because importing matplotlib would import numpy
# before netCDF4. poligrain requires netCDF4 to be imported before numpy (see
# src/poligrain/__init__.py), otherwise a "numpy.ndarray size changed" warning
# is raised, which pytest turns into an error via filterwarnings.
os.environ.setdefault("MPLBACKEND", "Agg")
