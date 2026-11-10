"""Shared pytest fixtures and configuration for the test suite."""

import matplotlib as mpl

# Use a non-interactive backend so that tests do not require a display or
# Tkinter/Tcl, which is not reliably available on CI runners (e.g. Windows).
mpl.use("Agg")
