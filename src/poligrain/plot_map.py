from __future__ import annotations

import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize


def scatter_lines(
    x0: np.ndarray | npt.ArrayLike | float,
    x1: np.ndarray | npt.ArrayLike | float,
    y1: np.ndarray | npt.ArrayLike | float,
    y0: np.ndarray | npt.ArrayLike | float,
    s: float = 3,
    c: str | np.array = "C0",
    line_style: str = "-",
    pad_width: float = 1,
    cap_style: str = "round",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: (str | Colormap) = "viridis",
    ax: (matplotlib.axes.Axes | None) = None,
) -> LineCollection:
    if ax is None:
        _, ax = plt.subplots()

    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)
    x1 = np.atleast_1d(x1)
    y1 = np.atleast_1d(y1)

    data = None if isinstance(c, str) else c

    if data is None:
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            linewidth=s,
            linestyles=line_style,
            color=c,
        )

    else:
        if vmax is None:
            vmax = np.nanmax(data)
        if vmin is None:
            vmin = np.nanmin(data)
        norm = Normalize(vmin=vmin, vmax=vmax)
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            norm=norm,
            cmap=cmap,
            linewidth=s,
            linestyles=line_style,
            capstyle=cap_style,
            path_effects=[
                pe.Stroke(linewidth=s + pad_width, foreground="k", capstyle=cap_style),
                pe.Normal(),
            ],
        )
        lines.set_array(data)

    ax.add_collection(lines)
    # This is required because x and y bounds are not adjusted after adding the `lines`.
    ax.autoscale()

    return lines


def plot_lines(
    cmls: (xr.Dataset | xr.DataArray),
    vmin: (float | None) = None,
    vmax: (float | None) = None,
    cmap: (str | Colormap) = "turbo",
    line_color: str = "k",
    line_width: float = 1,
    pad_width: float = 1,
    ax: (matplotlib.axes.Axes | None) = None,
) -> LineCollection:
    """Plot paths of line-based sensors like CMLs.

    If a `xarray.Dataset` is passed, the paths are plotted using the defined
    `line_color`. If a `xarray.DataArray` is passed its content is used to
    color the lines based on `cmap`, `vmin` and `vmax`. The `xarray.DataArray`
    has to be 1D with one entry per line.

    Parameters
    ----------
    cmls : xr.Dataset  |  xr.DataArray
        The line-based sensors data with coordinates defined according to the
        OPENSENSE data format conventions.
    vmin : float  |  None, optional
        Minimum value of colormap, by default None.
    vmax : float  |  None, optional
        Maximum value of colormap, by default None.
    cmap : str  |  Colormap, optional
        A matplotlib colormap either as string or a `Colormap` object, by default "turbo".
    line_color : str, optional
        The color of the lines when plotting based on a `xarray.Dataset`, by default "k".
    line_width : float, optional
        The width of the lines. In case of coloring lines with a `cmap`, this is the
        width of the colored line, which is extend by `pad_width` with a black outline.
        By default 1.
    pad_width : float, optional
        The width of the outline in black around the lines colored via a `cmap`, by default 1.
    ax : matplotlib.axes.Axes  |  None, optional
        A `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.

    Returns
    -------
    LineCollection

    """
    if ax is None:
        _, ax = plt.subplots()

    try:
        color_data = cmls.data
        if len(color_data.shape) != 1:
            msg = (
                f"If you pass an xarray.DataArray it has to be 1D, with the length of the "
                f"cml_id dimension. You passed in something with shape {color_data.shape}"
            )
            raise ValueError(msg)
    except AttributeError:
        color_data = line_color

    return scatter_lines(
        x0=cmls.site_0_lon.values,
        y0=cmls.site_0_lat.values,
        x1=cmls.site_1_lon.values,
        y1=cmls.site_1_lat.values,
        s=line_width,
        c=color_data,
        pad_width=pad_width,
        line_style="-",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
