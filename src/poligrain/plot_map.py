from __future__ import annotations

import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize


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
            linewidth=line_width,
            color=line_color,
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
            linewidth=line_width,
            linestyles="solid",
            capstyle="round",
            path_effects=[
                pe.Stroke(
                    linewidth=line_width + pad_width, foreground="k", capstyle="round"
                ),
                pe.Normal(),
            ],
        )
        lines.set_array(data)

    ax.add_collection(lines)
    # This is required because x and y bounds are not adjusted after adding the `lines`.
    ax.autoscale()

    return lines


@xr.register_dataset_accessor("plg")
class PoligrainDatasetAccessor:
    """Accessor for functionality of poligrain"""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def plot_cmls(
        self,
        line_color: str = "k",
        line_width: float = 1,
        pad_width: float = 1,
        ax: (matplotlib.axes.Axes | None) = None,
    ) -> LineCollection:
        """Plot paths of CMLs and color the lines based on data.

        Parameters
        ----------
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
        return plot_lines(
            self._dataset,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            ax=ax,
        )


@xr.register_dataarray_accessor("plg")
class PoligrainDataArrayAccessor:
    """Accessor for functionality of poligrain"""

    def __init__(self, data_array: xr.DataArray):
        self._data_array = data_array

    def plot_cmls(
        self,
        vmin: (float | None) = None,
        vmax: (float | None) = None,
        cmap: (str | Colormap) = "turbo",
        line_color: str = "k",
        line_width: float = 1,
        pad_width: float = 1,
        ax: (matplotlib.axes.Axes | None) = None,
    ) -> LineCollection:
        """Plot paths of CMLs and color the lines based on data.

        Parameters
        ----------
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
        return plot_lines(
            self._data_array,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            ax=ax,
        )
