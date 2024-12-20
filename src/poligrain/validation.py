"""Functions for calculating verification metrics and making (scatter) plots."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap


def plot_hexbin(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    ref_thresh: float = 0.0,
    est_thresh: float = 0.0,
    gridsize: (int | tuple[int, int]) = 45,
    cmap: (str | Colormap) = "viridis",
    colorbar: bool = True,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> PolyCollection:
    """Scatter density plot of reference values against the estimated values.

    This function compares the estimated to reference values equal to, or above, a
    given threshold, on a point to point basis. The threshold can be different for
    the reference and estimated values. Note: it does not take any information on
    temporal resolution or aggregation into account, so these have to already match
    in the input numpy arrays.


    Parameters
    ----------
    reference : npt.ArrayLike
        The reference values to be plotted on the x-axis.
    estimate : npt.ArrayLike
        The estimated values to be plotted on the y-axis.
    ref_thresh : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    est_thresh : float, optional
        All values >= threshold in estimate are taken
        into account. By default 0.0., i.e. no threshold.
    grid_size : int, optional
        Number of hexagons in x-direction, or, if a tuple, number of hexagons in x- and
        y-direction. By default 45.
    cmap : str  |  Colormap, optional
        A matplotlib colormap either as string or a `Colormap` object,
        by default "viridis".
    colorbar : bool, optional
        Whether to draw a colorbar, by default True.
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

    # filter out values strictly less than the threshold
    thresh_idx = (reference >= ref_thresh) | (estimate >= est_thresh)

    # set maximum plot extent to the nearest 10
    max_extent = np.ceil(np.nanmax([reference, estimate]) / 10) * 10

    # scatter density plot
    hx = ax.hexbin(
        reference[thresh_idx],
        estimate[thresh_idx],
        bins="log",
        mincnt=1,
        gridsize=gridsize,
        extent=(0, max_extent, 0, max_extent),
        **kwargs,
    )

    # add a 1:1 line
    ax.plot([0, max_extent], [0, max_extent], color="gray", alpha=0.7)

    # set a discrete colorbar and make the ticks match segments
    if colorbar is True:
        oom = int(np.ceil(np.log10(hx.get_clim()[1])))
        hx.set_clim([1, 10**oom])
        hx.set_cmap(plt.get_cmap(cmap, oom * 2))  # 2 colors per order of magnitude
        cbar = plt.colorbar(hx, ax=ax)
        cbar.set_label("count")

    # set axis labels to make clear what is plotted. Can be changed outside the function
    ax.set_xlabel("Reference rainfall rate \nalong the CML path [mm/h]")
    ax.set_ylabel("CML rainfall rate [mm/h]")

    return hx


def calculate_rainfall_metrics(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    ref_thresh: float = 0.0,
    est_thresh: float = 0.0,
) -> dict[str, float]:
    """Calculate verification metrics for rainfall estimation.

    This function calculates verification metrics based on reference and estimated
    rainfall rates, equal to or above a given threshold. NaNs are excluded.
    The threshold is applied to all metric calculations so the metrics essentially match
    the data plotted in the scatter plots (validation.plot_hexbin). Note thate the units 
    of 'R_sum...' and 'R_mean...' depend on the units of the input arrays.
    Metrics include:
    - Pearson correlation coefficient (r)
    - Coefficient of variation (CV)
    - Root mean square error (RMSE)
    - Mean absolute error (MAE)
    - Percent bias (PBias)

    Parameters
    ----------
    reference : npt.ArrayLike
        Rainfall reference.
    estimate : npt.ArrayLike
        Estimated rainfall.
    ref_thresh : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    est_thresh : float, optional
        All values >= threshold in estimates are taken
        into account. By default 0.0., i.e. no threshold.

    Returns
    -------
    dict[str, float]
        A dictionary containing key value pairs of the verification metrics.
    """
    assert reference.shape == estimate.shape

    # select all value pairs if one or both are NaN to calculate metrics
    nan_idx = np.isnan(reference) | np.isnan(estimate)

    N_all = len(reference)
    N_nan = np.sum(nan_idx)
    N_nan_ref = np.sum(np.isnan(reference))
    N_nan_est = np.sum(np.isnan(estimate))

    # exclude NaNs
    reference = reference[~nan_idx]
    estimate = estimate[~nan_idx]

    # apply threshold and filter out values strictly less than the threshold
    thresh_idx = (reference >= ref_thresh) | (estimate >= est_thresh)
    reference_ge_thresh = reference[thresh_idx]
    estimate_ge_thresh = estimate[thresh_idx]

    assert reference.shape == estimate.shape

    # calculate metrics: r, CV, RMSE, MAE, PBIAS
    pearson_correlation = np.corrcoef(reference_ge_thresh, estimate_ge_thresh)
    coefficient_of_variation = np.std(estimate_ge_thresh - reference_ge_thresh) / np.mean(reference_ge_thresh)
    root_mean_square_error = np.sqrt(np.mean((estimate_ge_thresh - reference_ge_thresh) ** 2))
    mean_absolute_error = np.mean(np.abs(estimate_ge_thresh - reference_ge_thresh))
    percent_bias = (np.mean(estimate_ge_thresh - reference_ge_thresh) / np.mean(reference_ge_thresh)) * 100  # %

    # calculate mean rainfall
    R_mean_reference = reference_ge_thresh.mean()
    R_mean_estimate = estimate_ge_thresh.mean()
    R_sum_reference = reference_ge_thresh.sum()
    R_sum_estimate = estimate_ge_thresh.sum()

    # calculate wet-dry statistics based on threshold
    reference_wet = reference > ref_thresh
    reference_dry = np.logical_not(reference_wet)
    estimate_wet = estimate > est_thresh
    estimate_dry = np.logical_not(estimate_wet)

    N_false_wet = (reference_dry & estimate_wet).sum()
    N_dry = reference_dry.sum()
    false_wet_ratio = N_false_wet / float(N_dry)

    N_missed_wet = (reference_wet & estimate_dry).sum()
    N_wet = reference_wet.sum()
    missed_wet_ratio = N_missed_wet / float(N_wet)

    false_wet_r_mean = estimate[reference_dry & estimate_wet].mean()
    missed_wet_r_mean = reference[reference_wet & estimate_dry].mean()

    return {
        "reference_rainfall_threshold": ref_thresh,
        "estimates_rainfall_threshold": est_thresh,
        "pearson_correlation_coefficient": pearson_correlation[0, 1],
        "coefficient_of_variation": coefficient_of_variation,
        "root_mean_square_error": root_mean_square_error,
        "mean_absolute_error": mean_absolute_error,
        "percent_bias": percent_bias,
        "reference_mean_rainfall": R_mean_reference,
        "estimated_mean_rainfall": R_mean_estimate,
        "reference_rainfall_sum": R_sum_reference,
        "estimated_rainfall_sum": R_sum_estimate,
        "false_wet_ratio": false_wet_ratio,
        "missed_wet_ratio": missed_wet_ratio,
        "false_wet_rainfall_rate": false_wet_r_mean,
        "missed_wet_rainfall_rate": missed_wet_r_mean,
        "N_all": N_all,
        "N_nan": N_nan,
        "N_nan_reference": N_nan_ref,
        "N_nan_estimate": N_nan_est,
    }


def calculate_wet_dry_metrics(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
) -> dict[str, float]:
    """Calculate verification metrics for wet-dry classification.

    This function calculates verification metrics based on binary classification of wet
    and dry intervals in the reference and estimated data. If the input array is not
    boolean but contains rainfall values, a threshold of 0 is set, and any value > 0 
    is considered 'wet'. NaNs are excluded from the calculation in the function.
    Metrics include:
    - Matthews correlation coefficient (MCC)

    Parameters
    ----------
    reference : npt.ArrayLike
        Boolean array of reference rainfall with 'wet' being True.
    estimate : npt.ArrayLike
        Boolean array of estimated rainfall with 'wet' being True.

    Returns
    -------
    dict[str, float]
        A dictionary containing key value pairs of the verification metrics.
    """
    assert reference.shape == estimate.shape

    # select all value pairs if one or both are NaN to calculate metrics
    nan_idx = np.isnan(reference) | np.isnan(estimate)

    N_all = len(reference)
    N_nan = np.sum(nan_idx)
    N_nan_ref = np.sum(np.isnan(reference))
    N_nan_est = np.sum(np.isnan(estimate))

    # exclude NaNs
    reference = reference[~nan_idx]
    estimate = estimate[~nan_idx]

    # force bool type
    reference = reference > 0
    estimate = estimate > 0

    # calculate the MCC
    N_tp = ((reference == True) & (estimate == True)).sum()
    N_tn = np.sum((reference == False) & (estimate == False))
    N_fp = np.sum((reference == False) & (estimate == True))
    N_fn = np.sum((reference == True) & (estimate == False))

    N_wet_ref = np.sum(reference == True)
    N_dry_ref = np.sum(reference == False)

    true_wet_ratio = N_tp / N_wet_ref
    true_dry_ratio = N_tn / N_dry_ref
    false_wet_ratio = N_fp / N_dry_ref
    missed_wet_ratio = N_fn / N_wet_ref

    a = np.sqrt(N_tp + N_fp)
    b = np.sqrt(N_tp + N_fn)
    c = np.sqrt(N_tn + N_fp)
    d = np.sqrt(N_tn + N_fn)

    matthews_correlation_coefficient = ((N_tp * N_tn) - (N_fp * N_fn)) / (a * b * c * d)

    # if estimated has zero/false values only 'inf' would be returned, but 0 is
    # preferred
    if np.isinf(matthews_correlation_coefficient):
        matthews_correlation_coefficient = 0
    if np.nansum(estimate) == 0:
        matthews_correlation_coefficient = 0

    return {
        "matthews_correlation_coefficient": matthews_correlation_coefficient,
        "true_wet_ratio": true_wet_ratio,
        "true_dry_ratio": true_dry_ratio,
        "false_wet_ratio": false_wet_ratio,
        "missed_wet_ratio": missed_wet_ratio,
        "N_dry_reference": N_dry_ref,
        "N_wet_reference": N_wet_ref,
        "N_true_wet": N_tp,
        "N_true_dry": N_tn,
        "N_false_wet": N_fp,
        "N_missed_wet": N_fn,
        "N_all": N_all,
        "N_nan": N_nan,
        "N_nan_reference": N_nan_ref,
        "N_nan_estimate": N_nan_est,
    }


def print_metrics_table(metrics: dict[str, float]) -> None:
    """
    Print pretty table with the verification metrics.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing the verification metrics.
        Keys are the metric names and values are the metric values.
        Supports both rainfall and wet-dry metrics.

    Returns
    -------
    None
    """
    # Calculate the width for the metric column
    max_metric_length = max(len(name) for name in metrics.keys())
    metric_column_width = max_metric_length + 2  # Add padding for readability
    value_column_width = 15  # Fixed width for the value column

    # Create table header elements
    table_lines = []
    separator = f"+{'-' * metric_column_width}+{'-' * value_column_width}+"
    header = (
        f"| {'Metric'.center(metric_column_width-1)}"
        f"| {'Value'.center(value_column_width-1)}|"
    )
    double_separator = f"+{'=' * metric_column_width}+{'=' * value_column_width}+"

    # Add header
    table_lines.append("Verification metrics:")
    table_lines.append("")
    table_lines.append(double_separator)
    table_lines.append(header)
    table_lines.append(double_separator)

    # Add metrics to the table
    for k, val in metrics.items():
        metric_name = k.replace("_", " ")
        metric_name = metric_name.capitalize()

        row = (
            f"| {metric_name.ljust(metric_column_width-1)}"
            f"| {f'{np.round(val, 3)}'.center(value_column_width - 1)}|"
        )
        table_lines.append(row)
        table_lines.append(separator)

    return print("\n".join(table_lines))


def plot_confusion_matrix_distributions(
    reference: npt.ArrayLike,
    predicted: npt.ArrayLike,
    ref_thresh: float = 0.0,
    pred_thr: float = 0.0,
    N_bins: int = 101,
    bin_type: str = "linear",
    ax: (matplotlib.axes.Axes | None) = None,
) -> dict[str, list]:
    """Plot the distributions of the confusion matrix.

    The true positives and false positives are based on the predicted data. The false
    negatives are based on the reference data.
    The extent of the bins runs from 0.01 to 100 mm/h.

    Parameters
    ----------
    reference : npt.ArrayLike
        Rainfall reference.
    predicted : npt.ArrayLike
        Predicted rainfall.
    ref_thresh : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    pred_thr : float, optional
        All values >= threshold in predicted are taken
        into account. By default 0.0., i.e. no threshold.
    N_bins : int, optional
        Number of bins to use in the histograms. By default 101.
    bin_type : str, optional
        Type of binning to use. Either "linear" or "log". By default "linear".
    ax : matplotlib.axes.Axes  |  None, optional
    An `Axes` object on which to plot. If not supplied, a new figure with an
    `Axes` will be created. By default None.

    Returns
    -------
    PolyCollection
    """
    if bin_type == "linear":
        bins = np.linspace(0, 100, N_bins)
        x_values = np.linspace(0, 100, N_bins - 1)
    elif bin_type == "log":
        bins = np.logspace(
            -2, 2, num=N_bins, endpoint=True, base=10, dtype=None, axis=0
        )
    x_values = np.logspace(
        -2, 2, num=N_bins - 1, endpoint=True, base=10, dtype=None, axis=0
    )

    # initiate the histograms
    tp_mask = np.logical_and(predicted >= pred_thr, reference >= ref_thresh)
    tp, _ = np.histogram(predicted[tp_mask], bins=N_bins)

    fp_mask = np.logical_and(predicted >= pred_thr, reference < ref_thresh)
    fp, _ = np.histogram(predicted[fp_mask], bins=N_bins)

    fn_mask = np.logical_and(predicted < pred_thr, reference >= ref_thresh)
    fn, _ = np.histogram(reference[fn_mask], bins=bins)

    # plot the histograms
    if ax is None:
        _, ax = plt.subplots()

    x_values = x_values

    l1 = ax.plot(x_values, (tp) / ds_cmls_size, color="C0", linewidth=0.5)
    p1 = ax.fill_between(
        x_values, (tp) / ds_cmls_size, alpha=0.3, color="C0", label="TP"
    )

    l2 = ax.plot(x_values, (fp) / ds_cmls_size, color="C2", linewidth=0.5)
    p2 = ax.fill_between(
        x_values, (fp) / ds_cmls_size, alpha=0.3, color="C2", label="FP"
    )

    l3 = ax.plot(x_values, (fn) / ds_cmls_size, color="C3", linewidth=0.5)
    p3 = ax.fill_between(
        x_values, (fn) / ds_cmls_size, alpha=0.3, color="C3", label="FN"
    )

    ax.set_xscale("log")
    ax.axvspan(0, ref_thresh, alpha=0.1, color="grey", label="DRY")
    ax.legend()

    return {"lines": [l1, l2, l3], "fills": [p1, p2, p3]}
