"""Functions for calculating verification metrics and making (scatter) plots."""

from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap
from matplotlib.patches import StepPatch


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
    """Scatter density plot of reference against the estimated rainfall values.

    This function compares the estimated to reference values equal to, or above, a
    given threshold, on a point to point basis. The threshold can be different for
    the reference and estimated values. Note: it does not take any information on
    temporal resolution or alignment into account, so these have to already match
    in the input arrays!


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
    assert reference.shape == estimate.shape

    if ax is None:
        _, ax = plt.subplots()

    # filter out values strictly less than the threshold
    thresh_idx = (reference < ref_thresh) | (estimate < est_thresh)

    # set maximum plot extent to the nearest 10
    max_extent = np.ceil(np.nanmax([reference, estimate]) / 10) * 10

    # scatter density plot
    hx = ax.hexbin(
        reference[np.logical_not(thresh_idx)],
        estimate[np.logical_not(thresh_idx)],
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
    """Verification metrics for rainfall estimation.

    This function calculates verification metrics based on reference and estimated
    rainfall, equal to or above a given threshold. The threshold is
    applied to all metric calculations so the metrics essentially match the data
    plotted in the scatter plots (validation.plot_hexbin). This means values below the
    threshold are excluded from the calculations. NaNs are also excluded. Note: the
    units of the mean rainfall in the output will depend on the units of the input
    arrays (typically mm/h or mm).
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
    thresh_idx = (reference < ref_thresh) | (estimate < est_thresh)
    reference_ge_thresh = reference[np.logical_not(thresh_idx)]
    estimate_ge_thresh = estimate[np.logical_not(thresh_idx)]

    assert reference.shape == estimate.shape

    # calculate metrics: r, CV, RMSE, MAE, PBIAS
    pearson_correlation = np.corrcoef(reference_ge_thresh, estimate_ge_thresh)
    coefficient_of_variation = np.std(
        estimate_ge_thresh - reference_ge_thresh
    ) / np.mean(reference_ge_thresh)
    root_mean_square_error = np.sqrt(
        np.mean((estimate_ge_thresh - reference_ge_thresh) ** 2)
    )
    mean_absolute_error = np.mean(np.abs(estimate_ge_thresh - reference_ge_thresh))
    percent_bias = (
        np.mean(estimate_ge_thresh - reference_ge_thresh) / np.mean(reference_ge_thresh)
    ) * 100  # %

    # calculate mean rainfall
    R_mean_reference = reference_ge_thresh.mean()
    R_mean_estimate = estimate_ge_thresh.mean()

    # calculate rainfall statistics of confusion matrix
    reference_wet = reference > ref_thresh
    reference_dry = np.logical_not(reference_wet)
    estimate_wet = estimate > est_thresh
    estimate_dry = np.logical_not(estimate_wet)

    FP_r_mean = estimate[reference_dry & estimate_wet].mean()
    FN_r_mean = reference[reference_wet & estimate_dry].mean()

    return {
        "ref_thresh": ref_thresh,
        "est_thresh": est_thresh,
        "pearson_correlation_coefficient": pearson_correlation[0, 1],
        "coefficient_of_variation": coefficient_of_variation,
        "root_mean_square_error": root_mean_square_error,
        "mean_absolute_error": mean_absolute_error,
        "percent_bias": percent_bias,
        "reference_mean_rainfall": R_mean_reference,
        "estimate_mean_rainfall": R_mean_estimate,
        "false_positive_mean_rainfall": FP_r_mean,
        "false_negative_mean_rainfall": FN_r_mean,
        "N_all": N_all,
        "N_nan": N_nan,
        "N_nan_ref": N_nan_ref,
        "N_nan_est": N_nan_est,
    }


def calculate_wet_dry_metrics(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    ref_thresh: float = 0.0,
    est_thresh: float = 0.0,
) -> dict[str, float]:
    """Verification metrics for wet-dry classification.

    This function calculates verification metrics based on reference and estimated
    rainfall, equal to or above a given threshold. The input arrays are turned
    into a boolean by considering any value >= the threshold as 'wet' (True), and else
    'dry' (False). Unlike the the validation.calculate_rainfall_metrics() function the
    values below the threshold are not excluded in the calculations, but set to 'dry'
    (False). Default threshold values in the reference and estimates is 0. NaNs
    are excluded from the calculation in the function.
    Metrics include:
    - Matthews correlation coefficient (MCC)

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

    # force bool type
    reference = reference >= ref_thresh
    estimate = estimate >= est_thresh

    assert reference.shape == estimate.shape

    # calculate the MCC
    N_tp = ((reference is True) & (estimate is True)).sum()
    N_tn = np.sum((reference is False) & (estimate is False))
    N_fp = np.sum((reference is False) & (estimate is True))
    N_fn = np.sum((reference is True) & (estimate is False))

    N_wet_ref = np.sum(reference is True)
    N_dry_ref = np.sum(reference is False)

    true_positive_ratio = N_tp / N_wet_ref
    true_negative_ratio = N_tn / N_dry_ref
    false_positive_ratio = N_fp / N_dry_ref
    false_negative_ratio = N_fn / N_wet_ref

    a = np.sqrt(N_tp + N_fp)
    b = np.sqrt(N_tp + N_fn)
    c = np.sqrt(N_tn + N_fp)
    d = np.sqrt(N_tn + N_fn)

    matthews_correlation_coefficient = ((N_tp * N_tn) - (N_fp * N_fn)) / (a * b * c * d)

    return {
        "matthews_correlation_coefficient": matthews_correlation_coefficient,
        "true_positive_ratio": true_positive_ratio,
        "true_negative_ratio": true_negative_ratio,
        "false_positive_ratio": false_positive_ratio,
        "false_negative_ratio": false_negative_ratio,
        "N_dry_ref": N_dry_ref,
        "N_wet_ref": N_wet_ref,
        "N_tp": N_tp,
        "N_tn": N_tn,
        "N_fp": N_fp,
        "N_fn": N_fn,
        "N_all": N_all,
        "N_nan": N_nan,
        "N_nan_ref": N_nan_ref,
        "N_nan_est": N_nan_est,
    }


def print_metrics_table(input_dict: dict[str, float], units: str = "mm/h") -> None:
    """
    Print pretty metrics table with the verification metrics.

    The descriptive names for the metrics in the table are looked up in the look-up
    dictionary provided in the function. The keys to access this dictionary are the
    same as the input dictionary.

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
    lookup_dict = {
        "ref_thresh": (
            "Reference rainfall threshold",
            lambda x: x["ref_thresh"],
            f"[{units}]",
        ),
        "est_thresh": (
            "Estimate rainfall threshold",
            lambda x: x["est_thresh"],
            f"[{units}]",
        ),
        "pearson_correlation_coefficient": (
            "Pearson correlation coefficient",
            lambda x: np.round(x["pearson_correlation_coefficient"], 3),
            "[-]",
        ),
        "coefficient_of_variation": (
            "Coefficient of variation",
            lambda x: np.round(x["coefficient_of_variation"], 3),
            "[-]",
        ),
        "root_mean_square_error": (
            "Root mean square error",
            lambda x: np.round(x["root_mean_square_error"], 3),
            f"[{units}]",
        ),
        "mean_absolute_error": (
            "Mean absolute error",
            lambda x: np.round(x["mean_absolute_error"], 3),
            f"[{units}]",
        ),
        "percent_bias": (
            "Percent bias",
            lambda x: np.round(x["percent_bias"], 3),
            "[%]",
        ),
        "reference_mean_rainfall": (
            "Mean reference rainfall",
            lambda x: np.round(x["reference_mean_rainfall"], 2),
            f"[{units}]",
        ),
        "estimate_mean_rainfall": (
            "Mean estimated rainfall",
            lambda x: np.round(x["estimate_mean_rainfall"], 2),
            f"[{units}]",
        ),
        "false_positive_mean_rainfall": (
            "False positive mean rainfall",
            lambda x: np.round(x["false_positive_mean_rainfall"], 2),
            f"[{units}]",
        ),
        "false_negative_mean_rainfall": (
            "False negative mean rainfall",
            lambda x: np.round(x["false_negative_mean_rainfall"], 2),
            f"[{units}]",
        ),
        "N_all": ("Total data points", lambda x: x["N_all"], "[-]"),
        "N_nan": ("Total nans", lambda x: x["N_nan"], "[-]"),
        "N_nan_ref": ("Number of nans in reference", lambda x: x["N_nan_ref"], "[-]"),
        "N_nan_est": ("Number of nans in estimate", lambda x: x["N_nan_est"], "[-]"),
        "N_dry_ref": (
            "Number of dry points in reference",
            lambda x: x["N_dry_ref"],
            "[-]",
        ),
        "N_wet_ref": (
            "Number of wet points in reference",
            lambda x: x["N_wet_ref"],
            "[-]",
        ),
        "N_tp": ("Number of true positives", lambda x: x["N_tp"], "[-]"),
        "N_tn": ("Number of true negatives", lambda x: x["N_tn"], "[-]"),
        "N_fp": ("Number of false positives", lambda x: x["N_fp"], "[-]"),
        "N_fn": ("Number of false negatives", lambda x: x["N_fn"], "[-]"),
        "matthews_correlation_coefficient": (
            "Matthews correlation coefficient",
            lambda x: np.round(x["matthews_correlation_coefficient"], 3),
            "[-]",
        ),
        "true_positive_ratio": (
            "True positive ratio",
            lambda x: np.round(x["true_positive_ratio"], 2),
            "[-]",
        ),
        "true_negative_ratio": (
            "True negative ratio",
            lambda x: np.round(x["true_negative_ratio"], 2),
            "[-]",
        ),
        "false_positive_ratio": (
            "False positive ratio",
            lambda x: np.round(x["false_positive_ratio"], 2),
            "[-]",
        ),
        "false_negative_ratio": (
            "False negative ratio",
            lambda x: np.round(x["false_negative_ratio"], 2),
            "[-]",
        ),
    }

    metrics = {
        key: (lookup_dict[key][0], lookup_dict[key][1](input_dict), lookup_dict[key][2])
        for key in input_dict
    }

    # Calculate the width for the metric column
    max_metric_length = max(len(name[0]) for name in metrics.values())
    metric_column_width = max_metric_length + 2  # Add padding for readability
    value_column_width = 15  # Fixed width for the value column
    unit_column_width = 10  # Fixed width for unit column

    # Create the table header
    table_lines = []
    separator = (
        f"+{'-' * metric_column_width}+"
        f"{'-' * value_column_width}+"
        f"{'-' * unit_column_width}+"
    )
    header = (
        f"| {'Metric'.center(metric_column_width-1)}"
        f"| {'Value'.center(value_column_width-1)}"
        f"| {'Units'.center(unit_column_width-1)}|"
    )
    double_separator = (
        f"+{'=' * metric_column_width}+"
        f"{'=' * value_column_width}+"
        f"{'=' * unit_column_width}+"
    )

    table_lines.append("Verification metrics:")
    table_lines.append("")
    table_lines.append(double_separator)
    table_lines.append(header)
    table_lines.append(double_separator)

    # Add metrics to the table
    for val in metrics.values():
        row = (
            f"| {val[0].ljust(metric_column_width-1)}"
            f"| {f'{val[1]}'.center(value_column_width - 1)}|"
            f" {val[2].center(unit_column_width-1)}|"
        )
        table_lines.append(row)
        table_lines.append(separator)

    return "\n".join(table_lines)


def plot_confusion_matrix_count(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    normalize_y: int = 1,
    ref_thresh: float = 0.0,
    est_thresh: float = 0.0,
    N_bins: int = 101,
    bin_type: str = "linear",
    bins: (npt.ArrayLike | None) = None,
    ax: (matplotlib.axes.Axes | None) = None,
) -> list(StepPatch):
    """Plot the count of the distributions of the confusion matrix.

    This function plots the distribution of the number of time intervals (counts) of
    true positives, false positives, and false negatives for N_bins (default 101) of
    rainfall intensity. The extent of the bins runs from 0.1 to 100 mm/h. Bins can be
    linear (default) or logarithmic. Supplying an array with custom bin edges is also
    possible using the kwarg 'bin' (which defaults to None). This will overwrite the
    hard coded linear and log bin options.
    By default the count refers to the total count in the entire data set. For an
    average count per sublink, and corresponding y-labels, provide the number of
    sublinks to normalize the y-axis with.

    Parameters
    ----------
    reference : npt.ArrayLike
        Rainfall reference.
    estimate : npt.ArrayLike
        Estimated rainfall.
    normalize_y : int, optional
        The number of sublinks with which the y-axis should be normalized. By default
        1, in which case the y-axis is not normalized and corresponds to the entire
        data set.
    ref_thresh : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    est_thresh : float, optional
        All values >= threshold in estimated are taken
        into account. By default 0.0., i.e. no threshold.
    N_bins : int, optional
        Number of bins to use in the histograms. By default 101.
    bin_type : str, optional
        Type of binning to use on the data. Either "linear" or "log".
        By default "linear".
    bins : npt.ArrayLike | None, optional
        Custom bin edges to use. If supplied, this will override `N_bins` and
        `bin_type`. By default None.
    ax : matplotlib.axes.Axes  |  None, optional
    An `Axes` object on which to plot. If not supplied, a new figure with an
    `Axes` will be created. By default None.

    Returns
    -------
    lines : list of StepPatches
    """
    assert reference.shape == estimate.shape

    if bins is not None:
        bins = np.asarray(bins)
    elif bin_type == "linear":
        bins = np.linspace(0.1, 100, N_bins)
    elif bin_type == "log":
        bins = np.logspace(
            -1, 2, num=N_bins, endpoint=True, base=10, dtype=None, axis=0
        )

    if normalize_y == 1:
        y_norm = 1
        y_label = "Count time intervals [-]"
    else:
        y_norm = normalize_y
        y_label = "Count time intervals \n per sublink [-]"

    # initiate the histograms
    tp_mask = np.logical_and(estimate >= est_thresh, reference >= ref_thresh)
    tp, _ = np.histogram(estimate[tp_mask], bins=bins)

    fp_mask = np.logical_and(estimate >= est_thresh, reference < ref_thresh)
    fp, _ = np.histogram(estimate[fp_mask], bins=bins)

    fn_mask = np.logical_and(estimate < est_thresh, reference >= ref_thresh)
    fn, _ = np.histogram(reference[fn_mask], bins=bins)

    # plot the histograms
    if ax is None:
        _, ax = plt.subplots()

    sp1 = ax.stairs(
        (tp) / y_norm,
        bins,
        color="tab:blue",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="TP",
    )

    sp2 = ax.stairs(
        (fp) / y_norm,
        bins,
        color="tab:green",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="FP",
    )

    sp3 = ax.stairs(
        (fn) / y_norm,
        bins,
        color="tab:orange",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="FN",
    )

    ax.set_xscale("log")
    ax.axvspan(0, ref_thresh, alpha=0.1, color="grey", label="DRY")

    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel("Rainfall Rate [mm/h]")

    ax.legend()

    return [sp1, sp2, sp3]


def plot_confusion_matrix_sum(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    time_interval: int,
    normalize_y: int = 1,
    ref_thresh: float = 0.0,
    est_thresh: float = 0.0,
    N_bins: int = 101,
    bin_type: str = "linear",
    bins: (npt.ArrayLike | None) = None,
    ax: (matplotlib.axes.Axes | None) = None,
) -> list(StepPatch):
    """Plot the rainfall sum of the distributions of the confusion matrix.

    This function plots the distribution of the rainfall sum of true positives, false
    positives, and false negatives for N_bins (default 101) of rainfall intensity.
    The extent of the bins runs from 0.1 to 100 mm/h. Bins can be linear (default) or
    logarithmic. Supplying an array with custom bin edges is also possible using the
    kwarg 'bin' (which defaults to None). This will overwrite the hard coded linear and
    log bin options.
    By default the sum refers to the total sum of the entire data set. For an average
    rainfall sum per sublink, and corresponding y-labels, provide the number of
    sublinks to normalize the y-axis with.

    Parameters
    ----------
    reference : npt.ArrayLike
        Rainfall reference.
    estimate : npt.ArrayLike
        Estimated rainfall.
    time_interval : int
        Time interval of the data, in minutes.
    normalize_y : int, optional
        The number of sublinks with which the y-axis should be normalized. By default
        1, in which case the y-axis is not normalized and corresponds to the entire
        data set.
    ref_thresh : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    est_thresh : float, optional
        All values >= threshold in estimated are taken
        into account. By default 0.0., i.e. no threshold.
    N_bins : int, optional
        Number of bins to use in the histograms. By default 101.
    bin_type : str, optional
        Type of binning to use on the data. Either "linear" or "log".
        By default "linear".
    bins : npt.ArrayLike | None, optional
        Custom bin edges to use. If supplied, this will override `N_bins` and
        `bin_type`. By default None.
    ax : matplotlib.axes.Axes  |  None, optional
    An `Axes` object on which to plot. If not supplied, a new figure with an
    `Axes` will be created. By default None.

    Returns
    -------
    lines : list of StepPatches
    """
    assert reference.shape == estimate.shape

    if bins is not None:
        bins = np.asarray(bins)
        bin_cent = (bins[:-1] + bins[1:]) / 2
    elif bin_type == "linear":
        bins = np.linspace(0.1, 100, N_bins)
        bin_cent = (bins[:-1] + bins[1:]) / 2
    elif bin_type == "log":
        bins = np.logspace(
            -1, 2, num=N_bins, endpoint=True, base=10, dtype=None, axis=0
        )
        bin_cent = (bins[:-1] + bins[1:]) / 2
    else:
        msg = "unsupported bin_type, must be 'linear' or 'log'"
        raise ValueError(msg)

    if normalize_y == 1:
        y_norm = 1
        y_label = "Rainfall amount [mm]"
    else:
        y_norm = normalize_y
        y_label = "Rainfall amount \nper sublink [mm]"

    # initiate the histograms
    tp_mask = np.logical_and(estimate >= est_thresh, reference >= ref_thresh)
    tp, _ = np.histogram(estimate[tp_mask], bins=bins)

    fp_mask = np.logical_and(estimate >= est_thresh, reference < ref_thresh)
    fp, _ = np.histogram(estimate[fp_mask], bins=bins)

    fn_mask = np.logical_and(estimate < est_thresh, reference >= ref_thresh)
    fn, _ = np.histogram(reference[fn_mask], bins=bins)

    # plot the histograms
    if ax is None:
        _, ax = plt.subplots()

    rate_to_sum = time_interval / 60  # convert rate to sum

    sp1 = ax.stairs(
        (bin_cent * rate_to_sum) * ((tp) / y_norm),
        bins,
        color="tab:blue",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="TP",
    )

    sp2 = ax.stairs(
        (bin_cent * rate_to_sum) * ((fp) / y_norm),
        bins,
        color="tab:green",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="FP",
    )

    sp3 = ax.stairs(
        (bin_cent * rate_to_sum) * ((fn) / y_norm),
        bins,
        color="tab:orange",
        linewidth=0.5,
        fill=True,
        alpha=0.3,
        label="FN",
    )

    ax.set_xscale("log")
    ax.axvspan(0, ref_thresh, alpha=0.1, color="grey", label="DRY")

    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel("Rainfall Rate [mm/h]")

    ax.legend()

    return [sp1, sp2, sp3]
