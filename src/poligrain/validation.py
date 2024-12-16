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
    predicted: npt.ArrayLike,
    ref_thr: float = 0.0,
    pred_thr: float = 0.0,
    gridsize: (int | tuple[int, int]) = 45,
    cmap: (str | Colormap) = "viridis",
    colorbar: bool = True,
    ax: (matplotlib.axes.Axes | None) = None,
    **kwargs,
) -> PolyCollection:
    """Scatter density plot of reference values against the predicted values.

    This function compares the predicted to reference values equal to, or above, a
    given threshold, on a point to point basis. The threshold can be different for
    the reference and predicted values. Note: it does not take any information on
    temporal resolution or aggregation into account, so these have to already match
    in the input numpy arrays.


    Parameters
    ----------
    reference : npt.ArrayLike
        The reference values to be plotted on the x-axis.
    predicted : npt.ArrayLike
        The predicted values to be plotted on the y-axis.
    ref_thr : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    pred_thr : float, optional
        All values >= threshold in predicted are taken
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
    thresh_idx = (reference >= ref_thr) | (predicted >= pred_thr)

    # set maximum plot extent to the nearest 10
    max_extent = np.ceil(np.nanmax([reference, predicted]) / 10) * 10

    # scatter density plot
    hx = ax.hexbin(
        reference[thresh_idx],
        predicted[thresh_idx],
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
    predicted: npt.ArrayLike,
    ref_thr: float = 0.0,
    pred_thr: float = 0.0,
) -> dict[str, float]:
    """Calculate verification metrics for rainfall estimation.

    This function calculates verification metrics based on reference and predicted
    rainfall rates, equal to or above a given threshold. NaNs are excluded.
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
    predicted : npt.ArrayLike
        Predicted rainfall.
    ref_thr : float, optional
        All values >= threshold in reference are taken
        into account. By default 0.0, i.e. no threshold.
    pred_thr : float, optional
        All values >= threshold in predicted are taken
        into account. By default 0.0., i.e. no threshold.

    Returns
    -------
    dict[str, float]
        A dictionary containing key value pairs of the verification metrics.
    """
    assert reference.shape == predicted.shape

    # select all value pairs if one or both are NaN to calculate metrics
    nan_idx = np.isnan(reference) | np.isnan(predicted)

    N_all = len(reference)
    N_nan = np.sum(nan_idx)

    # exclude NaNs
    reference = reference[~nan_idx]
    predicted = predicted[~nan_idx]

    # select pairs in reference or prediction strictly less than the threshold
    thresh_idx = (reference >= ref_thr) | (predicted >= pred_thr)
    reference = reference[thresh_idx]
    predicted = predicted[thresh_idx]

    # metrics: r, CV, RMSE, MAE, PBIAS
    pearson_correlation = np.corrcoef(reference, predicted)
    coefficient_of_variation = np.std(predicted - reference) / np.mean(reference)
    root_mean_square_error = np.sqrt(np.mean((predicted - reference) ** 2))
    mean_absolute_error = np.mean(np.abs(predicted - reference))
    percent_bias = (np.mean(predicted - reference) / np.mean(reference)) * 100  # %

    # fix potential division by zero by replacing 'inf' with 0
    if np.isinf(coefficient_of_variation):
        coefficient_of_variation = 0.0
    if np.isinf(percent_bias):
        percent_bias = 0.0

    # general rainfall statistics
    R_mean_reference = reference.mean()
    R_mean_predicted = predicted.mean()
    # R_sum_reference = reference.sum()
    # R_sum_predicted = predicted.sum()

    # wet-dry statistics based on threshold
    reference_wet = reference >= ref_thr
    reference_dry = np.logical_not(reference_wet)
    predicted_wet = predicted >= pred_thr
    predicted_dry = np.logical_not(predicted_wet)

    N_false_wet = (reference_dry & predicted_wet).sum()
    N_dry = reference_dry.sum()
    false_wet_rate = N_false_wet / float(N_dry)

    # if N_dry is zero, returning false_wet_rate=0 is preferred over 'inf' or 'nan'
    if np.isinf(false_wet_rate) or np.isnan(false_wet_rate):
        false_wet_rate = 0.0

    N_missed_wet = (reference_wet & predicted_dry).sum()
    N_wet = reference_wet.sum()
    missed_wet_rate = N_missed_wet / float(N_wet)

    # if N_wet is zero, returning missed_wet_rate=0 is preferred over 'inf' or 'nan'
    if np.isinf(missed_wet_rate) or np.isnan(missed_wet_rate):
        missed_wet_rate = 0.0

    false_wet_rainfall_rate = predicted[reference_dry & predicted_wet].mean()
    missed_wet_rainfall_rate = reference[reference_wet & predicted_dry].mean()

    # to avoid nans in the output if no false wet or dry values are present;
    # returns 0 instead

    false_wet_rainfall_rate = np.nan_to_num(false_wet_rainfall_rate)
    missed_wet_rainfall_rate = np.nan_to_num(missed_wet_rainfall_rate)

    return {
        "Reference_rainfall_threshold": ref_thr,
        "Predicted_rainfall_threshold": pred_thr,
        "Pearson_correlation coefficient": pearson_correlation[0, 1],
        "Coefficient_of_variation": coefficient_of_variation,
        "Root_mean_square_error": root_mean_square_error,
        "Mean_absolute_error": mean_absolute_error,
        "Percent_bias": percent_bias,
        "R_mean_reference": R_mean_reference,
        "R_mean_predicted": R_mean_predicted,
        # "R_sum_reference": R_sum_reference,
        # "R_sum_predicted": R_sum_predicted,
        "false_wet_rate": false_wet_rate,
        "missed_wet_rate": missed_wet_rate,
        "false_wet_rainfall_rate": false_wet_rainfall_rate,
        "missed_wet_rainfall_rate": missed_wet_rainfall_rate,
        "N_all": N_all,
        "N_nan": N_nan,
    }


def calculate_wet_dry_metrics(
    reference: npt.ArrayLike,
    predicted: npt.ArrayLike,
) -> dict[str, float]:
    """Calculate verification metrics for wet-dry classification.

    This function calculates verification metrics based on binary classification of wet
    and dry intervals in the reference and prediction data. Any value > 0 is considered
    'wet'. NaNs are excluded from the calculation in the function.
    Metrics include:
    - Matthews correlation coefficient (MCC)

    Parameters
    ----------
    reference : npt.ArrayLike
        Rainfall reference.
    predicted : npt.ArrayLike
        Predicted rainfall.

    Returns
    -------
    dict[str, float]
        A dictionary containing key value pairs of the verification metrics.
    """
    assert reference.shape == predicted.shape

    # select all value pairs if one or both are NaN to calculate metrics
    nan_idx = np.isnan(reference) | np.isnan(predicted)

    reference = reference[~nan_idx]
    predicted = predicted[~nan_idx]

    # calculate the MCC
    reference = reference > 0
    predicted = predicted > 0

    N_tp = np.sum((reference is True) & (predicted is True))
    N_tn = np.sum((reference is False) & (predicted is False))
    N_fp = np.sum((reference is False) & (predicted is True))
    N_fn = np.sum((reference is True) & (predicted is False))

    N_wet_ref = np.sum(reference is True)
    N_dry_ref = np.sum(reference is False)

    true_wet_rate = N_tp / N_wet_ref
    true_dry_rate = N_tn / N_dry_ref
    false_wet_rate = N_fp / N_dry_ref
    missed_wet_rate = N_fn / N_wet_ref

    a = np.sqrt(N_tp + N_fp)
    b = np.sqrt(N_tp + N_fn)
    c = np.sqrt(N_tn + N_fp)
    d = np.sqrt(N_tn + N_fn)

    matthews_correlation_coefficient = ((N_tp * N_tn) - (N_fp * N_fn)) / (a * b * c * d)

    # if predicted has zero/false values only 'inf' would be returned, but 0 is more
    # favorable
    if np.isinf(matthews_correlation_coefficient):
        matthews_correlation_coefficient = 0
    if np.nansum(predicted) == 0:
        matthews_correlation_coefficient = 0

    return {
        "matthews_correlation_coefficient": matthews_correlation_coefficient,
        "true_wet_rate": true_wet_rate,
        "true_dry_rate": true_dry_rate,
        "false_wet_rate": false_wet_rate,
        "missed_wet_rate": missed_wet_rate,
        "N_dry_reference": N_dry_ref,
        "N_wet_reference": N_wet_ref,
        "N_true_wet": N_tp,
        "N_true_dry": N_tn,
        "N_false_wet": N_fp,
        "N_missed_wet": N_fn,
    }


# def print_verification_metrics(
#     metrics: dict[str, float]
# ) -> None:
#

# def plot_confusion_matrix_distributions(
#         reference: npt.ArrayLike,
#         predicted: npt.ArrayLike,
#         ref_thr: float = 0.0,
#         pred_thr: float = 0.0,
#         N_bins: int = 101,
#         bin_type: str = "linear",
#         ax: (matplotlib.axes.Axes | None) = None
# ) -> PolyCollection:
#     """Plot the distributions of the confusion matrix.

#     The true positives and false positives are based on the predicted data. The false
#     negatives are based on the reference data.
#     The extent of the bins runs from 0.01 to 100 mm/h.

#     Parameters
#     ----------
#     reference : npt.ArrayLike
#         Rainfall reference.
#     predicted : npt.ArrayLike
#         Predicted rainfall.
#     ref_thr : float, optional
#         All values >= threshold in reference are taken
#         into account. By default 0.0, i.e. no threshold.
#     pred_thr : float, optional
#         All values >= threshold in predicted are taken
#         into account. By default 0.0., i.e. no threshold.
#     N_bins : int, optional
#         Number of bins to use in the histograms. By default 101.
#     bin_type : str, optional
#         Type of binning to use. Either "linear" or "log". By default "linear".
# ax : matplotlib.axes.Axes  |  None, optional
#     An `Axes` object on which to plot. If not supplied, a new figure with an
#     `Axes` will be created. By default None.

#     Returns
#     -------
#     PolyCollection
#     """

#     if bin_type == "linear":
#         bins = np.linspace(0, 100, N_bins)
#         x_values = np.linspace(0, 100, N_bins-1)
#     elif bin_type == "log":
#         bins = np.logspace(
#         -2,
#         2,
#         num=N_bins,
#         endpoint=True,
#         base=10,
#         dtype=None,
#         axis=0
#     )
#     x_values = np.logspace(
#         -2,
#         2,
#         num=N_bins-1,
#         endpoint=True,
#         base=10,
#         dtype=None,
#         axis=0
#     )

#     # initiate the histograms
#     tp_mask = np.logical_and(predicted >= pred_thr, reference >= ref_thr)
#     tp, _ = np.histogram(
#         predicted[tp_mask],
#         bins=N_bins
#     )

#     fp_mask = np.logical_and(predicted >= pred_thr, reference < ref_thr)
#     fp, _ = np.histogram(
#         predicted[fp_mask],
#         bins=N_bins
#     )

#     fn_mask = np.logical_and(predicted < pred_thr, reference >= ref_thr)
#     fn, _ = np.histogram(
#         reference[fn_mask],
#         bins=bins
#     )

#     # plot the histograms
#     if ax is None:
#         _, ax = plt.subplots()

#     x_values = x_values

#     ax.plot(x_values, (tp) / ds_cmls_size, color="C0", linewidth=0.5)
#     ax.fill_between(x_values, (tp) / ds_cmls_size, alpha=0.3, color="C0", label="TP")

#     ax.plot(x_values, (fp) / ds_cmls_size, color="C2", linewidth=0.5)
#     ax.fill_between(x_values, (fp) / ds_cmls_size, alpha=0.3, color="C2", label="FP")

#     ax.plot(x_values, (fn) / ds_cmls_size, color="C3", linewidth=0.5)
#     ax.fill_between(x_values, (fn) / ds_cmls_size, alpha=0.3, color="C3", label="FN")

#     ax.set_xscale("log")
#     ax.axvspan(0, ref_thr, alpha=0.1, color="grey", label="DRY")

#     return ax
