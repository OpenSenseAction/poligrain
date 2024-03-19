"""xarray Accessors."""

import matplotlib.axes
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap

from . import plot_map


@xr.register_dataset_accessor("plg")
class PoligrainDatasetAccessor:
    """Accessor for functionality of poligrain."""

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
            The color of the lines when plotting based on a `xarray.Dataset`,
            by default "k".
        line_width : float, optional
            The width of the lines. In case of coloring lines with a `cmap`, this is the
            width of the colored line, which is extend by `pad_width` with
            a black outline. By default 1.
        pad_width : float, optional
            The width of the outline in black around the lines colored via a `cmap`,
            by default 1.
        ax : matplotlib.axes.Axes  |  None, optional
            A `Axes` object on which to plot. If not supplied, a new figure with
            an `Axes` will be created. By default None.

        Returns
        -------
        LineCollection

        """
        return plot_map.plot_lines(
            self._dataset,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            ax=ax,
        )


@xr.register_dataarray_accessor("plg")
class PoligrainDataArrayAccessor:
    """Accessor for functionality of poligrain."""

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
            A matplotlib colormap either as string or a `Colormap` object,
            by default "turbo".
        line_color : str, optional
            The color of the lines when plotting based on a `xarray.Dataset`,
            by default "k".
        line_width : float, optional
            The width of the lines. In case of coloring lines with a `cmap`, this is the
            width of the colored line, which is extend by `pad_width` with
            a black outline. By default 1.
        pad_width : float, optional
            The width of the outline in black around the lines colored via a `cmap`,
            by default 1.
        ax : matplotlib.axes.Axes  |  None, optional
            A `Axes` object on which to plot. If not supplied, a new figure with
            an `Axes` will be created. By default None.

        Returns
        -------
        LineCollection

        """
        return plot_map.plot_lines(
            self._data_array,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            ax=ax,
        )
