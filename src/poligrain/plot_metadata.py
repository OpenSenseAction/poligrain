"""Functions for plotting meta data of line-based sensors."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import Colormap
from matplotlib.container import BarContainer


def plot_len_vs_freq(
    length: xr.DataArray,
    frequency: xr.DataArray,
    marker_color: str = "k",
    marker_size: float = 10,
    grid: bool = True,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> PathCollection:
    """Scatter plot of length vs. frequency of line-based sensors like CMLs.

    The frequency 'xr.DataArray' should have dimensions (cml_id, sublink_id), and the
    length 'xr.DataArray' should have dimensions (cml_id) according to OPENSENSE data
    formats. Length is expected to be in meters and frequency in megahertz.


    Parameters
    ----------
    length : xr.DataArray
        Path length of line-based sensors, units in meters, according to the OPENSENSE
        data format.
    frequency : xr.DataArray
        Frequency of line-based sensors, units in megahertz, according to the OPENSENSE
        data format.
    marker_color : str, optional
        Color of the markers. By default "k".
    marker_size : float, optional
        Size of the markers. By default 10.
    grid : bool, optional
        Add major grid lines. By default True.
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.
    **kwargs
        Optional keyword arguments to pass to the `scatter` function.

    Returns
    -------
    PathCollection
    """
    if ax is None:
        _, ax = plt.subplots()

    # divide frequency and length by 1000 to convert to km and GHz
    len_values = length.broadcast_like(frequency).to_numpy() / 1000
    freq_values = frequency.to_numpy() / 1000

    # scatter plot
    scatter = ax.scatter(
        len_values, freq_values, color=marker_color, marker=".", s=marker_size, **kwargs
    )

    # add axis labels
    ax.set_xlabel("Length (km)")
    ax.set_ylabel("Frequency (GHz)")

    # add grid
    if grid:
        ax.grid()
    return scatter


def plot_len_vs_freq_hexbin(
    length: xr.DataArray,
    frequency: xr.DataArray,
    cmap: (str | Colormap) = "viridis",
    gridsize: (int | tuple[int, int]) = 45,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> PolyCollection:
    """Scatter density plot of length vs. frequency of line-based sensors like CMLs.

    The frequency 'xr.DataArray' should have dimensions (cml_id, sublink_id), and the
    length 'xr.DataArray' should have dimensions (cml_id) according to OPENSENSE data
    formats.


    Parameters
    ----------
    length : xr.DataArray
        Path length of line-based sensors, according to the OPENSENSE data format
        conventions in meters.
    frequency : xr.DataArray
        Frequency of line-based sensors, according to the OPENSENSE data format
        conventions in megahertz.
    cmap : str  |  Colormap, optional
        A matplotlib colormap either as string or a `Colormap` object,
        by default "viridis".
    grid_size : int, optional
        Number of hexagons in x-direction, or, if a tuple, number of hexagons in x- and
        y-direction. By default 45.
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.
    **kwargs
        Optional keyword arguments to pass to the `hexbin` function.

    Returns
    -------
    PolyCollection
    """
    if ax is None:
        _, ax = plt.subplots()

    # divide frequency and length by 1000 to convert to km and GHz
    len_values = length.broadcast_like(frequency).values / 1000  # noqa: PD011
    freq_values = frequency.values / 1000  # noqa: PD011

    # scatter density plot
    hexbin = ax.hexbin(
        len_values, freq_values, mincnt=1, cmap=cmap, gridsize=gridsize, **kwargs
    )

    # add axis labels
    ax.set_xlabel("Length (km)")
    ax.set_ylabel("Frequency (GHz)")

    return hexbin


def plot_len_vs_freq_jointplot(
    length: xr.DataArray,
    frequency: xr.DataArray,
    marker_color: str = "k",
    marker_size: float = 10,
    grid: bool = True,
    bin_width_len: float = 1,
    bin_width_freq: float = 1,
    axes: (list[matplotlib.axes.Axes] | None) = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[PathCollection],
    PathCollection,
    np.ndarray,
    np.ndarray,
    list[PathCollection],
]:
    """Scatter plot of path length vs. frequency with histograms as margin plots.

    This function mimics Seaborn's `jointplot` function, but relies only on Matplotlib.
    It creates a scatter plot of path length vs. frequency as the main plot with the
    distribution of each as variable as marginal histograms.

    Parameters
    ----------
    length : xr.DataArray
        Path length of line-based sensors, according to the OPENSENSE data format
        conventions in meters.
    frequency : xr.DataArray
        Frequency of line-based sensors, according to the OPENSENSE data format
        conventions in megahertz.
    marker_color : str, optional
        Color of the markers in the main plot. By default "k".
    marker_size : int, optional
        Size of the markers in the main plot. By default 10.
    grid : bool, optional
        Add major grid lines to the main plot. By default True.
    bin_width_len : float, optional
        Width of the bins (kms) for the path length margin plot. By default 1 km bins.
    bin_width_freq : float, optional
        Width of the bins (GHz) for the frequency margin plot. By default 1 GHz bins.
    axes : list[matplotlib.axes.Axes]  |  None, optional
        A list of `Axes` objects in order of a figure with 2x2 subplots. I.e. [top left,
        top right, bottom left, bottom right]. Defaults to None. If not supplied, a new
        figure with four `Axes` will be created.  Note that the top right subplot will
        be turned off.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[PathCollection],
    PathCollection,
    np.ndarray, np.ndarray, list[PathCollection]]
    """
    if axes is None:
        _, axes = plt.subplots(
            2,
            2,
            gridspec_kw={
                "hspace": 0.05,
                "wspace": 0.05,
                "width_ratios": [5, 1],
                "height_ratios": [1, 5],
            },
        )
    ax = axes.flatten()

    # turn off top right subplot
    ax[1].axis("off")

    # divide frequency and length by 1000 to convert to km and GHz
    # and flatten arrays to plot histograms with a single color
    len_values = length.broadcast_like(frequency).values.flatten() / 1000  # noqa: PD011
    freq_values = frequency.values.flatten() / 1000  # noqa: PD011

    # -----------------------------------
    # MAIN SCATTER PLOT
    # -----------------------------------
    scatter = ax[2].scatter(len_values, freq_values, color=marker_color, s=marker_size)
    ax[2].set_xlabel("Length [km]")
    ax[2].set_ylabel("Frequency [GHz]")

    # Add gridlines
    if grid:
        ax[2].grid(True, linestyle="-", color="lightgray")
        ax[2].set_axisbelow(True)

    # Remove black tick marks but keep labels
    ax[2].tick_params(axis="y", length=0)
    ax[2].tick_params(axis="x", length=0)

    # Adapt spines for Seaborn-look
    for spine in ax[2].spines.values():
        spine.set_color("darkgrey")

    # -----------------------------------
    # TOP MARGIN HISTOGRAM (x-axis)
    # -----------------------------------
    bin_width_x = bin_width_len
    bins_len = np.arange(
        np.floor(len_values.min()),
        np.ceil(len_values.max()) + bin_width_x,
        bin_width_x,
    )

    hist_x, bins_x, patches_x = ax[0].hist(
        len_values, bins=bins_len, color="lightgray", edgecolor="white"
    )

    # Remove x-ticks
    ax[0].tick_params(axis="x", bottom=False, labelbottom=False)

    # Remove y-ticks but keep labels
    ax[0].tick_params(axis="y", length=0)

    # Add more y-tick labels
    ax[0].yaxis.set_major_locator(plt.MaxNLocator(3))

    # Label y-axis
    ax[0].set_ylabel("Count")

    # Add grid: solid vertical, dashed horizontal
    ax[0].grid(axis="x", linestyle="-", color="lightgray")
    ax[0].grid(axis="y", linestyle="--", color="lightgray")

    # Adapt spines for Seaborn-look
    for spine in ["left", "right", "top"]:
        ax[0].spines[spine].set_visible(False)
    ax[0].spines["bottom"].set_color("darkgray")

    # Align limits of marginals with main scatter plot
    ax[0].set_xlim(ax[2].get_xlim())
    ax[3].set_ylim(ax[2].get_ylim())

    # -----------------------------------
    # RIGHT MARGIN HISTOGRAM (y-axis)
    # -----------------------------------
    bin_width_y = bin_width_freq
    bins_freq = np.arange(
        np.floor(freq_values.min()),
        np.ceil(freq_values.max()) + bin_width_y,
        bin_width_y,
    )

    hist_y, bins_y, patches_y = ax[3].hist(
        freq_values,
        bins=bins_freq,
        orientation="horizontal",
        color="lightgray",
        edgecolor="white",
    )

    # Remove y-ticks and labels
    ax[3].tick_params(axis="y", left=False, labelleft=False)

    # Remove x-tick marks but keep labels
    ax[3].tick_params(axis="x", length=0)

    # Add more x-tick labels
    ax[3].xaxis.set_major_locator(plt.MaxNLocator(3))

    # Move x-ticks to top
    ax[3].xaxis.tick_top()
    ax[3].xaxis.set_label_position("top")
    ax[3].set_xlabel("Count")

    # Add grid: solid horizontal, dashed vertical
    ax[3].grid(axis="y", linestyle="-", color="lightgray")
    ax[3].grid(axis="x", linestyle="--", color="lightgray")

    # Adapt spines for Seaborn-look
    for spine in ["right", "top", "bottom"]:
        ax[3].spines[spine].set_visible(False)
    ax[3].spines["left"].set_color("darkgray")

    return hist_x, bins_x, patches_x, scatter, hist_y, bins_y, patches_y


def plot_distribution(
    length: xr.DataArray,
    frequency: xr.DataArray,
    variable: str = "length",
    percentage: bool = False,
    bins: (int | np.ndarray) = 20,
    color: str = "grey",
    edgecolor: str = "black",
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[PathCollection]]:
    """Histogram with distribution of path length or frequency of CMLs.

    This produces only one plot, either for length or frequency, depending on what has
     been passed to the argument 'variable'.

    The frequency 'xr.DataArray' should have dimensions (cml_id, sublink_id), and the
    length 'xr.DataArray' should have dimensions (cml_id) according to OPENSENSE data
    formats.

    Parameters
    ----------
    length : xr.DataArray
        Path length of line-based sensors, according to the OPENSENSE data format
        conventions in meters.
    frequency : xr.DataArray
        Frequency of line-based sensors, according to the OPENSENSE data format
        conventions in megahertz.
    variable : str
        Variable to plot. Either 'length' or 'frequency'. By default 'length'.
    percentage : bool
        If True, then the number of sublinks per bin are plotted as a percentage,
        otherwise they are plotted as a count. By default True.
    bins : int  |  np.ndarray, optional
        Number of bins or bin edges. By default 10.
    color : str, optional
        Color of the histogram. By default "grey".
    edgecolor : str, optional
        Color of the edges. By default "black".
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.
    **kwargs
        Optional keyword arguments to pass to the `hist` function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[PathCollection]]
    """
    if ax is None:
        _, ax = plt.subplots()

    hist, bins, patches = None, None, None

    # divide frequency and length by 1000 to convert to km and GHz
    len_values = length.broadcast_like(frequency).values.flatten() / 1000  # noqa: PD011
    freq_values = frequency.values.flatten() / 1000  # noqa: PD011

    if variable == "length":
        if percentage:
            # assign weights and plot histogram
            w = np.ones_like(len_values) * 100 / len(len_values)
            hist, bins, patches = ax.hist(
                len_values,
                bins=bins,
                weights=w,
                color=color,
                edgecolor=edgecolor,
                **kwargs,
            )
            ax.set_ylabel("Percentage (%)")

        else:
            hist, bins, patches = ax.hist(
                len_values, bins=bins, color=color, edgecolor=edgecolor, **kwargs
            )
            ax.set_ylabel("Count (nr. of sublinks)")

        # add x-axis labels
        ax.set_xlabel("Length (km)")

    elif variable == "frequency":
        if percentage:
            # assign weights and plot histogram
            w = np.ones_like(freq_values) * 100 / len(freq_values)
            hist, bins, patches = ax.hist(
                freq_values,
                bins=bins,
                weights=w,
                color=color,
                edgecolor=edgecolor,
                **kwargs,
            )
            ax.set_ylabel("Percentage (%)")

        else:
            hist, bins, patches = ax.hist(
                freq_values, bins=bins, color=color, edgecolor=edgecolor, **kwargs
            )
            ax.set_ylabel("Count (nr. of sublinks)")

        # add x-axis labels
        ax.set_xlabel("Frequency (GHz)")

    return hist, bins, patches


def plot_polarization(
    polarization: xr.DataArray,
    colors: (list[str, str, str] | None) = None,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> BarContainer:
    """Bar graph with the count of CML polarization. Either HH, VV or HV.

    The polarization 'xr.DataArray' should have dimensions (cml_id, sublink_id).

    Parameters
    ----------
    Polarization : xr.DataArray
        Polarization of line-based sensors, according to the OPENSENSE data format
        conventions written as full word strings i.e. 'vertical', 'horizontal'.
    colors : list, optional
        List of three strings with the colors of the three bars 'HH', 'VV', 'HV'.
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.

    Returns
    -------
    BarContainer
    """
    if ax is None:
        _, ax = plt.subplots()

    # Initialize counts
    count_hh = 0
    count_vv = 0
    count_hv = 0

    for i in range(polarization.sizes["cml_id"]):
        # Iterate through each cml_id to count the polarization combinations
        sublink_pol = polarization[:, i].values  # noqa: PD011

        # Change all strings to lower case to avoid count errors due to capitalization
        sublink_pol_lower = [pol.lower() for pol in sublink_pol]

        # Check for strings that are allowed by the OpenSense data format
        if all(pol in ("h", "horizontal") for pol in sublink_pol_lower):
            count_hh += 1
        elif all(pol in ("v", "vertical") for pol in sublink_pol_lower):
            count_vv += 1
        elif any(pol in ("h", "horizontal") for pol in sublink_pol_lower) and any(
            pol in ("v", "vertical") for pol in sublink_pol_lower
        ):
            count_hv += 1

    # Data for plotting
    labels = ["HH", "VV", "HV"]
    counts = [count_hh, count_vv, count_hv]

    # Plot the bar graph
    bars = ax.bar(labels, counts, color=colors, **kwargs)
    ax.set_xlabel("Polarization")
    ax.set_ylabel("Count (nr. of CMLs)")

    return bars


def plot_availability_distribution(
    dataset: xr.Dataset,
    variable: str = "rsl",
    bins: (int | np.ndarray) = 10,
    color: str = "grey",
    edgecolor: str = "black",
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[PathCollection]]:
    """Histogram with distribution of data avaibility per cml.

    Plots availability as a percentage of the total data set length.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with variables named according to OPENSENSE data format.
    variable : str
        Variable to derive the availability from. For example 'rsl' or 'tsl'.
        By default 'rsl'.
    percentage : bool
        If True, then the number of sublinks per bin are plotted as a percentage,
        otherwise they are plotted as a count. By default True.
    bins : int  |  np.ndarray, optional
        Number of bins or bin edges. By default 10.
    color : str, optional
        Color of the histogram. By default "grey".
    edgecolor : str, optional
        Color of the edges. By default "black".
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.
    **kwargs
        Optional keyword arguments to pass to the `scatter` function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[PathCollection]]
    """
    if ax is None:
        _, ax = plt.subplots()

    hist, bins, patches = None, None, None

    # Count valid (non-NaN) values over time
    valid_counts = dataset[variable].count(dim="time")

    # Total number of time steps
    total_counts = dataset.sizes["time"]

    # Compute percentage availability
    availability_pct = (valid_counts / total_counts) * 100

    # Flatten array to plot histograms with a single color
    availability_pct = availability_pct.values.flatten()  # noqa: PD011

    # assign weights and plot histogram
    w = np.ones_like(availability_pct) * 100 / len(availability_pct)
    hist, bins, patches = ax.hist(
        availability_pct,
        bins=bins,
        weights=w,
        color=color,
        edgecolor=edgecolor,
        **kwargs,
    )

    ax.set_xticks(np.arange(0, 110, 10))

    # add axes labels
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Data availability of cmls (%)")

    return hist, bins, patches


def plot_availability_time_series(
    dataset: xr.Dataset,
    variable: str = "rsl",
    show_links: str = "both",
    resample_to: (str | None) = None,
    mean_over: (str | None) = None,
    marker_color_sublinks: str = "k",
    marker_color_cmls: str = "grey",
    marker_size: float = 10,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> tuple(PathCollection, PathCollection):
    """Scatter plot of sublink and cml availability over time.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with variables named according to OPENSENSE data format.
    variable : str
        Variable to derive the availability from. For example 'rsl' or 'tsl'.
        By default 'rsl'.
    show_links : {'both', 'sublinks', 'cmls'}, optional
        Which links to show in the plot.
        - 'both': plot both sublinks and link paths (default)
        - 'sublinks': only plot sublinks
        - 'cmls': only plot link paths
    resample_to : (str | None)
        Optional interval to resample the availability in case the native frequency is
        too high.         Strings should follow xarray's resampling arguments,
        i.e 'D' for daily, 'H' for hourly. By default None.
    mean_over: (str | None)
        Optional period to take the mean over. Must be a dt attribute like 'hour',
        'dayofweek', 'month', etc. By default None.
    marker_color_sublinks : str, optional
        Color of the markers for the sublink time series. By default "k".
    marker_color_cmls : str, optional
        Color of the markers for the cml time series. By default "grey".
    marker_size : float, optional
        Size of the markers. By default 10.
    ax : matplotlib.axes.Axes  |  None, optional
        An `Axes` object on which to plot. If not supplied, a new figure with an `Axes`
        will be created. By default None.
    **kwargs
        Optional keyword arguments to pass to the `scatter` function.

    Returns
    -------
    tuple(PathCollection, PathCollection)
    """
    if ax is None:
        _, ax = plt.subplots()

    availability_bool = dataset[variable].notnull()  # noqa: PD004

    # True if any sublink for a given cml is valid at that time step
    cmls_available = availability_bool.any(dim="sublink_id")

    # Count how many cml_ids are available per time step
    num_cmls = cmls_available.sum(dim="cml_id")

    # Number of sublinks available at each time step
    num_sublinks = availability_bool.sum(dim=("cml_id", "sublink_id"))

    time_array = availability_bool["time"]

    # Optionally resample to e.g. daily means
    if resample_to is not None:
        num_sublinks = num_sublinks.resample(time=resample_to).mean()
        num_cmls = num_cmls.resample(time=resample_to).mean()

        time_array = (
            availability_bool["time"]
            .resample(time=resample_to)
            .mean()
            .time.dt.floor(resample_to)
        )

    # Optionally take the mean over a certain period, e.g. diurnal variation
    if mean_over is not None:
        period = getattr(dataset[variable]["time"].dt, mean_over.lower())
        num_sublinks = num_sublinks.groupby(period).mean()
        num_cmls = num_cmls.groupby(period).mean()
        time_array = np.unique(period.values)

    # Create scatter plot
    scatter_sublinks = scatter_cmls = None  # to avoid reference before assignment error

    if show_links in ("both", "sublinks"):
        scatter_sublinks = ax.scatter(
            time_array,
            num_sublinks,
            color=marker_color_sublinks,
            s=marker_size,
            label="sublinks",
            **kwargs,
        )

    if show_links in ("both", "cmls"):
        scatter_cmls = ax.scatter(
            time_array,
            num_cmls,
            color=marker_color_cmls,
            s=marker_size,
            label="link paths",
            **kwargs,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean nr. of available links per time interval")

    return scatter_sublinks, scatter_cmls
