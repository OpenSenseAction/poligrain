from __future__ import annotations

import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_lines(
    cmls: (xr.Dataset | xr.DataArray),
    linewidth: float = 1,
    ax: (matplotlib.axes.Axes | None) = None,
) -> LineCollection:
    """Plot CML paths"""
    if ax is None:
        _, ax = plt.subplots()

    try:
        data = cmls.data
        if len(data.shape) != 1:
            msg = f"If you pass an xarray.DataArray it has to be 1D, with the length of the cml_id dimension. You passed in something with shape {data.shape}"
            raise ValueError(msg)
    except AttributeError:
        data = None

    x0 = np.atleast_1d(cmls.site_0_lon.values)
    y0 = np.atleast_1d(cmls.site_0_lat.values)
    x1 = np.atleast_1d(cmls.site_1_lon.values)
    y1 = np.atleast_1d(cmls.site_1_lat.values)

    if data is None:
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            linewidth=linewidth,
        )

    else:
        vmax = np.nanmax(data)
        vmin = np.nanmin(data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        pad_width = 1
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            norm=norm,
            cmap="turbo",
            linewidth=linewidth,
            linestyles="solid",
            capstyle="round",
            path_effects=[
                pe.Stroke(
                    linewidth=linewidth + pad_width, foreground="k", capstyle="round"
                ),
                pe.Normal(),
            ],
        )
        lines.set_array(data)

    ax.add_collection(lines)
    ax.autoscale()

    return lines
