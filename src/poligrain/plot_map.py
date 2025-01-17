"""Functions for plotting."""

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
    x0: npt.ArrayLike | float,
    y0: npt.ArrayLike | float,
    x1: npt.ArrayLike | float,
    y1: npt.ArrayLike | float,
    s: float = 3,
    c: (str | npt.ArrayLike) = "C0",
    line_style: str = "-",
    pad_width: float = 0,
    pad_color: str = "k",
    cap_style: str = "round",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: (str | Colormap) = "viridis",
    ax: (matplotlib.axes.Axes | None) = None,
) -> LineCollection:
    """Plot lines as if you would use plt.scatter for points.

    Parameters
    ----------
    x0 : npt.ArrayLike | float
        x coordinate of start point of line
    y0 : npt.ArrayLike | float
        y coordinate of start point of line
    x1 : npt.ArrayLike | float
        x coordinate of end point of line
    y1 : npt.ArrayLike | float
        y coordinate of end point of line
    s : float, optional
        The width of the lines. In case of coloring lines with a `cmap`, this is the
        width of the colored line, which is extend by `pad_width` with colored outline
        using `pad_color`. By default 1.
    c : str  |  npt.ArrayLike, optional
        The color of the lines. If something array-like is passe, this data is used
        to color the lines based on the `cmap`, `vmin` and `vmax`. By default "C0".
    line_style : str, optional
        Line style as used by matplotlib, default is "-".
    pad_width : float, optional
        The width of the outline, i.e. edge width, around the lines, by default 0.
    pad_color: str, optional
        Color of the padding, i.e. the edge color of the lines. Default is "k".
    cap_style: str, optional
        Whether to have "round" or rectangular ("butt") ends of the lines.
        Default is "round".
    vmin : float  |  None, optional
        Minimum value of colormap, by default None.
    vmax : float  |  None, optional
        Maximum value of colormap, by default None.
    cmap : str  |  Colormap, optional
        A matplotlib colormap either as string or a `Colormap` object,
        by default "turbo".
    ax : matplotlib.axes.Axes  |  None, optional
        A `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.

    Returns
    -------
    LineCollection
        _description_
    """
    if ax is None:
        _, ax = plt.subplots()

    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)
    x1 = np.atleast_1d(x1)
    y1 = np.atleast_1d(y1)

    data = None if isinstance(c, str) else c

    if pad_width == 0:
        path_effects = None
    else:
        path_effects = [
            pe.Stroke(
                linewidth=s + pad_width, foreground=pad_color, capstyle=cap_style
            ),
            pe.Normal(),
        ]

    if data is None:
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            linewidth=s,
            linestyles=line_style,
            capstyle=cap_style,
            color=c,
            path_effects=path_effects,
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
            path_effects=path_effects,
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
    use_lon_lat=True,
    cmap: (str | Colormap) = "turbo",
    line_color: str = "C0",
    line_width: float = 1,
    pad_width: float = 0.5,
    pad_color: str = "k",
    line_style: str = "-",
    cap_style: str = "round",
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
        A matplotlib colormap either as string or a `Colormap` object,
        by default "turbo".
    line_color : str, optional
        The color of the lines when plotting based on a `xarray.Dataset`,
        by default "k".
    line_width : float, optional
        The width of the lines. In case of coloring lines with a `cmap`, this is the
        width of the colored line, which is extend by `pad_width` with a black outline.
        By default 1.
    pad_width : float, optional
        The width of the outline, i.e. edge width, around the lines, by default 0.
    pad_color: str, optional
        Color of the padding, i.e. the edge color of the lines. Default is "k".
    line_style : str, optional
        Line style as used by matplotlib, default is "-".
    cap_style: str, optional
        Whether to have "round" or rectangular ("butt") ends of the lines.
        Default is "round".
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
                f"If you pass an xarray.DataArray it has to be 1D, with the length of "
                f"the cml_id dimension. You passed in something with shape "
                f"{color_data.shape}"
            )
            raise ValueError(msg)
    except AttributeError:
        color_data = line_color

    if use_lon_lat:
        x0_name = "site_0_lon"
        x1_name = "site_1_lon"
        y0_name = "site_0_lat"
        y1_name = "site_1_lat"
    else:
        x0_name = "site_0_x"
        x1_name = "site_1_x"
        y0_name = "site_0_y"
        y1_name = "site_1_y"

    return scatter_lines(
        x0=cmls[x0_name].values,
        y0=cmls[y0_name].values,
        x1=cmls[x1_name].values,
        y1=cmls[y1_name].values,
        s=line_width,
        c=color_data,
        pad_width=pad_width,
        pad_color=pad_color,
        cap_style=cap_style,
        line_style=line_style,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )


def plot_plg(
    da_grid=None,
    da_cmls=None,
    da_gauges=None,
    vmin=None,
    vmax=None,
    cmap="turbo",
    ax=None,
    use_lon_lat=True,
    edge_color="k",
    edge_width=0.5,
    marker_size=20,
    kwargs_cmls_plot=None,
    kwargs_gauges_plot=None,
):
    """Plot point, line and grid data.

    The data to be plotted has to be provided as xr.DataArray conforming
    to our naming conventions. Data has to be for one selected time step.

    Data of the three different sources can be passed all at once, but one
    can also pass only one or two of them. `vmin`, `vmax` and `cmap` will be
    the same for all three data sources, but can be adjusted separately via
    `kwargs_cmls_plot` and `kwargs_gauges_plot`.

    Parameters
    ----------
    da_grid : xr.DataArray, optional
        2D gridded data (only one time step), typically from weather radar
    da_cmls : xr.DataArray, optional
        CML data (for one specific time step)
    da_gauges : xr.DataArray, optional
        Gauge data (for on specific time step)
    vmin : float, optional
        vmin for all three data sources, by default None. If set to None
        it will be derived individually for each data source when plotting.
    vmax : float, optional
        vmax for all three data sources, by default None. If set to None
        it will be derived individually for each data source when plotting.
    cmap : str, optional
        cmap for all three data sources, by default "turbo"
    ax : _type_, optional
        Axes object from matplotlib, by default None which will create a new
        figure and return the Axes object.
    use_lon_lat : bool, optional
        If set to True use lon-lat coordinates for plotting. If set to False
        use x-y coordinates (meant to be projected coordinates). By default True.
        Note that our data conventions enforce that lon-lat coordinates are provided,
        but projected coordinates might need to be generated first before plotting.
        This plotting function does not project data on the fly.
    edge_color : str, optional
        Edge color of points and lines, by default "k"
    edge_width : float, optional
        Width of edge line of points and lines, by default 0.5
    marker_size : int, optional
        Size of points and lines, by default 20. Note that the value is directly
        passed to plt.scatter for plotting points but for the width of the lines
        it is divided by 10 so that visually the have more or less the same size.
    kwargs_cmls_plot : dict or None, optional
        kwargs to be passed to the CML plotting function, by default None. See
        `plot_lines` for supported kwargs.
    kwargs_gauges_plot : dict or None, optional
        kwargs to be passed to plt.scatter, by default None.

    """
    if kwargs_cmls_plot is None:
        kwargs_cmls_plot = {}
    if kwargs_gauges_plot is None:
        kwargs_gauges_plot = {}

    if ax is None:
        _, ax = plt.subplots()

    if use_lon_lat:
        grid_x_name = "lon"
        grid_y_name = "lat"
        point_x_name = "lon"
        point_y_name = "lat"
    else:
        grid_x_name = "xs"
        grid_y_name = "ys"
        point_x_name = "x"
        point_y_name = "y"

    if da_grid is not None:
        da_grid.plot.pcolormesh(
            x=grid_x_name, y=grid_y_name, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax
        )
    if da_cmls is not None:
        plot_lines(
            cmls=da_cmls,
            vmin=kwargs_cmls_plot.pop("vmin", vmin),
            vmax=kwargs_cmls_plot.pop("vmax", vmax),
            cmap=kwargs_cmls_plot.pop("cmap", cmap),
            use_lon_lat=use_lon_lat,
            ax=ax,
            line_width=kwargs_cmls_plot.pop("line_width", marker_size / 10),
            pad_color=kwargs_cmls_plot.pop("edge_color", edge_color),
            pad_width=kwargs_cmls_plot.pop("edge_width", edge_width),
            **kwargs_cmls_plot,
        )
    if da_gauges is not None:
        ax.scatter(
            x=da_gauges[point_x_name],
            y=da_gauges[point_y_name],
            c=da_gauges.data,
            vmin=kwargs_gauges_plot.pop("vmin", vmin),
            vmax=kwargs_gauges_plot.pop("vmax", vmax),
            cmap=kwargs_gauges_plot.pop("cmap", cmap),
            s=kwargs_gauges_plot.pop("s", marker_size),
            edgecolors=kwargs_gauges_plot.pop("edge_color", edge_color),
            linewidths=kwargs_gauges_plot.pop("line_width", edge_width),
            zorder=2,
            **kwargs_gauges_plot,
        )
    return ax
