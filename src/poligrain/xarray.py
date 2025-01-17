"""xarray Accessors."""
from __future__ import annotations

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
        line_color: str = "C0",
        line_width: float = 1,
        pad_width: float = 0,
        pad_color: str = "k",
        line_style: str = "-",
        cap_style: str = "round",
        use_lon_lat: bool = True,
        ax: (matplotlib.axes.Axes | None) = None,
    ) -> LineCollection:
        """Plot paths of CMLs and color the lines based on data.

        See `plot_map.plot_lines` for description of the parameters

        """
        return plot_map.plot_lines(
            self._dataset,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            pad_color=pad_color,
            line_style=line_style,
            cap_style=cap_style,
            use_lon_lat=use_lon_lat,
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
        pad_width: float = 0,
        pad_color: str = "k",
        line_style: str = "-",
        cap_style: str = "round",
        use_lon_lat: bool = True,
        ax: (matplotlib.axes.Axes | None) = None,
    ) -> LineCollection:
        """Plot paths of CMLs and color the lines based on data.

        See `plot_map.plot_lines` for description of the parameters

        """
        return plot_map.plot_lines(
            self._data_array,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            line_color=line_color,
            line_width=line_width,
            pad_width=pad_width,
            pad_color=pad_color,
            line_style=line_style,
            cap_style=cap_style,
            use_lon_lat=use_lon_lat,
            ax=ax,
        )
