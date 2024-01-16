from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection


def plot_lines(cmls: xr.Dataset, linewidth: float = 1) -> LineCollection:
    """Plot CML paths"""
    _, ax = plt.subplots()

    x0 = np.atleast_1d(cmls.site_0_lon.values)
    y0 = np.atleast_1d(cmls.site_0_lat.values)
    x1 = np.atleast_1d(cmls.site_1_lon.values)
    y1 = np.atleast_1d(cmls.site_1_lat.values)

    lines = LineCollection(
        [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
        linewidth=linewidth,
    )
    ax.add_collection(lines)
    ax.autoscale()

    return lines
