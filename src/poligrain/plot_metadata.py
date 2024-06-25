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
        Optional keyword arguments to pass to the `scatter` function.

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
        Optional keyword arguments to pass to the `scatter` function.

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
