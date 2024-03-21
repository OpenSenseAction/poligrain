"""Functions for plotting meta data of line-based sensors."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.collections import PathCollection


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
