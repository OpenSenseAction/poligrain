"""Functions for plotting."""

from __future__ import annotations

import cartopy
import cartopy.crs
import cartopy.io.img_tiles as cimgt
import matplotlib.axes
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize
import cartopy.mpl.ticker as cticker


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
    data_crs: (cartopy.crs.Projection | None) = None
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
    data_crs : cartopy.crs.Projection | None, optional
        The coordinate reference system of the data provided. The default is None.
        In the default case cartopy.crs.PlateCarree will be used when plotting
        with a `ax` that is a `cartopy.mpl.geoaxes.GeoAxes`. When plotting with `ax`
        being a normal `matplotlib.axes.Axes` `data_crs` has to be None since the
        coordinate transformation it implies are not supported by matplotlib.

    Returns
    -------
    LineCollection
        _description_
    """
    if ax is None:
        ax = plt.axes()

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

    style_kwargs = {
        "linewidth": s,
        "linestyle": line_style,
        "capstyle": cap_style,
        "path_effects": path_effects,
    }

    # We only allow a data_crs if we have a GeoAxes. And only in this case we
    # add a `transform` kwarg that is passed to `LineCollection` because it
    # seems that `transform = None` results in an empty plot when passing it
    # to `LineCollection` with `ax` being a normal matplotilb `Axes` object.
    if isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if data_crs is None:
            data_crs = cartopy.crs.PlateCarree()
        style_kwargs['transform'] = data_crs
    elif data_crs is not None:
        msg = 'data_crs has to be None if `ax` is not a cartopy.mpl.geoaxes.GeoAxes'
        raise ValueError(msg)

    if data is None:
        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            color=c,
            **style_kwargs,
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
            **style_kwargs,
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
    background_map: (str | None) = None,
    projection: (cartopy.crs.Projection | None) = None
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
    background_map : str | None, optional
        Type of background map.
    

    Returns
    -------
    LineCollection

    """
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

    if ax is None:
        ax = set_up_axes(background_map=background_map, projection=projection)

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


def set_up_axes(background_map='stock', projection=None):
    if background_map == 'stock':
        if projection is None:
            projection=cartopy.crs.PlateCarree()
        ax = plt.axes(projection=projection)
        ax.stock_img()
    elif background_map == 'OSM':
         request = cimgt.OSM()
         ax = plt.axes(projection=request.crs)
         ax.add_image(request, 10)
    elif background_map == 'NE':
        if projection is None:
            projection=cartopy.crs.PlateCarree()
        ax = plt.axes(projection=projection)
        ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue")
        ax.add_feature(cartopy.feature.LAND, facecolor="lightgrey")
        ax.add_feature(cartopy.feature.LAKES, facecolor="lightblue", linewidth=0.00001)
        ax.add_feature(cartopy.feature.BORDERS, linewidth=0.3)
        ax.coastlines(resolution="10m", linewidth=0.3)
    elif background_map is None:
        ax = plt.axes()
    else:
        msg = f'unsuported value of background_map {background_map}'
        raise ValueError(msg)
    return ax


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
    line_color="k",
    point_color="k",
    add_colorbar=True,
    colorbar_label="",
    kwargs_cmls_plot=None,
    kwargs_gauges_plot=None,
):
    """Plot point, line and grid data.

    The data to be plotted has to be provided as xr.DataArray or xr.Dataset conforming
    to our naming conventions. Data has to be for one selected time step if provided
    as xr.DataArray. For points and lines providing data as xr.Dataset is allowed and
    then only locations of sensors are plotted with single color.

    Data of the three different sources can be passed all at once, but one
    can also pass only one or two of them. `vmin`, `vmax` and `cmap` will be
    the same for all three data sources, but can be adjusted separately via
    `kwargs_cmls_plot` and `kwargs_gauges_plot`.

    Parameters
    ----------
    da_grid : xr.DataArray, optional
        2D gridded data (only one time step), typically from weather radar
    da_cmls : xr.DataArray or xr.Dataset, optional
        CML data (for one specific time step) if passed as xr.DataArray. If
        passed as xr.Dataset only the locations will be plotted.
    da_gauges : xr.DataArray, optional
        Gauge data (for on specific time step) if passed as xr.DataArray. If
        passed as xr.Dataset only the locations will be plotted.
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
    line_color : str, optional
        Color of lines if `da_cmls` is provided as xr.Dataset, by default "k".
        If `da_cmls` is provided as xr.DataArray this is ignored and the cmap
        is applied for the colored lines.
    point_color : str, optional
       Color of points if `da_gauges` is provided as xr.Dataset, by default "k".
       If `da_gauges` is provided as xr.DataArray this is ignored and the cmap
       is applied for the colored points.
    add_colorbar : bool, optional
        If True adds a color bar to the plot, by default True.
    colorbar_label : str, optional
        Label for the color bar, by default ""
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
        grid_x_name = "x_grid"
        grid_y_name = "y_grid"
        point_x_name = "x"
        point_y_name = "y"

    plotted_objects_for_cmap = []
    if da_grid is not None:
        pc = da_grid.plot.pcolormesh(
            x=grid_x_name,
            y=grid_y_name,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            add_colorbar=False,
        )
        plotted_objects_for_cmap.append(pc)
    if da_cmls is not None:
        line_collection = plot_lines(
            cmls=da_cmls,
            vmin=kwargs_cmls_plot.pop("vmin", vmin),
            vmax=kwargs_cmls_plot.pop("vmax", vmax),
            cmap=kwargs_cmls_plot.pop("cmap", cmap),
            use_lon_lat=use_lon_lat,
            ax=ax,
            line_width=kwargs_cmls_plot.pop("line_width", marker_size / 10),
            line_color=kwargs_cmls_plot.pop("line_color", line_color),
            pad_color=kwargs_cmls_plot.pop("edge_color", edge_color),
            pad_width=kwargs_cmls_plot.pop("edge_width", edge_width),
            **kwargs_cmls_plot,
        )
        # only add line_collection in case we really want to apply a cmap
        # based on CML data, i.e. only if passed as xr.DataArray and
        # not if passed as xr.Dataset.
        if isinstance(da_cmls, xr.DataArray):
            plotted_objects_for_cmap.append(line_collection)
    if da_gauges is not None:
        if isinstance(da_gauges, xr.DataArray):
            point_collection = ax.scatter(
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
            plotted_objects_for_cmap.append(point_collection)
        elif isinstance(da_gauges, xr.Dataset):
            ax.scatter(
                x=da_gauges[point_x_name],
                y=da_gauges[point_y_name],
                c=point_color,
                s=kwargs_gauges_plot.pop("s", marker_size),
                edgecolors=kwargs_gauges_plot.pop("edge_color", edge_color),
                linewidths=kwargs_gauges_plot.pop("line_width", edge_width),
                zorder=2,
                **kwargs_gauges_plot,
            )
        else:
            msg = "`da_gauges` has to be xr.Dataset or xr.DataArray"
            raise ValueError(msg)
    if add_colorbar and len(plotted_objects_for_cmap) > 0:
        plt.colorbar(plotted_objects_for_cmap[0], ax=ax, label=colorbar_label)
    return ax


def scatter_lines_background(
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
    transform = None,
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
    #if ax is None: >>> discarded this, also in plot_lines. Hope this is not an issue.
    #    _, ax = plt.subplots()

    # Value for request: determines resolution of background map / size of text (e.g. city names on map; for OpenStreetMap and GoogleMaps only).
    ValueRequest = 10
    type_background_map="OSM"
    # When zooming in use larger values. Too large values may result in an error. Then use lower values.

    #Coordinates = "[-10, 30, 32.7, 73]"	# Plotting area: minimum longitude, maximum longitude, minimum latitude, maximum latitude. Plotting area for Europe.
    Coordinates = "[11.4, 12.7, 57.2, 58.1]"
    extent = list(map(float, Coordinates.strip('[]').split(',')))	# Automatically obtain list of floats of values for bounding box.
    projection = cartopy.crs.epsg(3035)     		# Projection (epsg code), for Natural Earth background only.
                                           # epsg:3035 = ETRS89 / ETRS-LAEA (suited for Europe). See https://epsg.io/ for EPSG codes per region.
                                           # cartopy.crs.PlateCarree() can work for the entire world.
    ColorLand = "lightgrey"                  	# Color of land surface.
    ColorOceanRiverLakes = "lightblue"      	# Color of oceans, seas, rivers and lakes.
    DrawCoastlines = "yes"    			# "yes" for drawing coastlines. Note that coastlines and country borders seem less accurate compared to OpenStreetMap and GoogleMap.
    DrawCountries = "yes"	  			# "yes" for drawing country borders.
    DrawLakelines = "yes"     			# "yes" for drawing lake lines.
    transform = cartopy.crs.PlateCarree()
    # No need to add extend anymore, except for Natural Earth (NE).


    # To use OpenStreetMap:
    if type_background_map=="OSM":
        request = cimgt.OSM()
        # Set map:
        ax = plt.axes(projection=request.crs)
        #ax.set_extent(extent)
        ax.add_image(request, ValueRequest)
    # To use Google Maps:
    if type_background_map=="GM":
        style = "street"    # Style of background map (only for GoogleMaps): "satellite", "street".
        request = cimgt.GoogleTiles(style=style)
        # Set map:
        ax = plt.axes(projection=request.crs)
        #ax.set_extent(extent)
        ax.add_image(request, ValueRequest)
    # To use Natural Earth map:
    if type_background_map=="NE":
        # Map settings (e.g. projection and extent of area):
        ax = plt.axes(projection=projection)
        ax.set_extent(extent)   # >>>note THAT extent is only needed for Natural Earth maps.


    if type_background_map=="NE":
       # Add natural earth features and borders in case of Natural Earth map:
       ax.add_feature(cartopy.feature.LAND, facecolor=ColorLand)
       ax.add_feature(cartopy.feature.OCEAN, facecolor=ColorOceanRiverLakes)
       ax.add_feature(cartopy.feature.LAKES, facecolor=ColorOceanRiverLakes, linewidth=0.00001,zorder=1)
       if DrawCountries=="yes":    
          ax.add_feature(cartopy.feature.BORDERS, linestyle="-", linewidth=0.3, zorder=2)
       if DrawCoastlines=="yes":
           ax.coastlines(resolution="10m", linewidth=0.3, zorder=2)
       if DrawLakelines=="yes":
          ax.add_feature(cartopy.feature.LAKES, edgecolor="black", linewidth=0.3, facecolor="none",zorder=2)


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

    #else:>>>> this was the reason lines were not plotted on the OSM / GM / NE map: >>> do we really need this functionality?
        #if vmax is None:
         #   vmax = np.nanmax(data)
        #if vmin is None:
         #   vmin = np.nanmin(data)
        #norm = Normalize(vmin=vmin, vmax=vmax)

     

        lines = LineCollection(
            [((x0[i], y0[i]), (x1[i], y1[i])) for i in range(len(x0))],
            #norm=norm,
            cmap=cmap,
            linewidths=s,
            linestyles=line_style,
            capstyle=cap_style,
            path_effects=path_effects,
            transform=transform,
        )
        lines.set_array(data)

  
    # 2.4 Settings for parallels & meridians.
    alpha_parallel_meridians = 0.5			# The alpha blending value, between 0 (transparent) and 1 (opaque) for parallels & meridians.
    line_style_parallels_meridans = "--"		# The line style for parallels & meridians.
    line_width = 1					# Line width of parallels & meridians.
    PlotParallelsMeridiansInFront = "yes"           # "yes" for drawing parallels & meridians in front of all other visualizations.

    #ax.set_axisbelow(False)  # does not work to put parallel & meridians in front
    ax.gridlines(color='black', alpha=alpha_parallel_meridians, linestyle=line_style_parallels_meridans, linewidth=line_width, draw_labels=True)    # wish: do set in front; make grid lines optional

    ax.add_collection(lines)
    # This is required because x and y bounds are not adjusted after adding the `lines`.
    #ax.autoscale()
    ##return lines
    #return plt.show()
    return plt.savefig("test.jpg", bbox_inches = "tight", dpi = 300)
