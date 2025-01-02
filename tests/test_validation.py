from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

import poligrain as plg


def test_plot_hexbin():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()

    # Call the function
    hx = plg.validation.plot_hexbin(radar_array, cmls_array, ax=ax)

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)

    plt.close("all")


def test_plot_hexbin_without_ax():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    # Call the function without providing an axis
    hx = plg.validation.plot_hexbin(radar_array, cmls_array)

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)

    plt.close("all")


def test_plot_hexbin_without_colorbar():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()

    # Call the function without colorbar
    hx = plg.validation.plot_hexbin(radar_array, cmls_array, colorbar=False, ax=ax)

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)

    plt.close("all")


def test_calculate_rainfall_metrics_with_thresholds():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, 1, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    metrics = plg.validation.calculate_rainfall_metrics(
        ref_array, est_array, ref_thresh=0.1, est_thresh=0.1
    )

    assert metrics["reference_rainfall_threshold"] == 0.1
    assert metrics["estimates_rainfall_threshold"] == 0.1
    assert np.isclose(
        metrics["pearson_correlation_coefficient"], -0.385, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["coefficient_of_variation"], 1.759, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["root_mean_square_error"], 0.803, atol=0.001, equal_nan=True
    )
    assert np.isclose(metrics["mean_absolute_error"], 0.676, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["percent_bias"], 2.189, atol=0.001, equal_nan=True)

    assert np.isclose(
        metrics["reference_mean_rainfall"], 0.456, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_mean_rainfall"], 0.456, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["reference_rainfall_sum"], 4.109, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_rainfall_sum"], 4.200, atol=0.001, equal_nan=True
    )

    assert np.isclose(metrics["false_wet_ratio"], 0.375, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["missed_wet_ratio"], 0.75, atol=0.001, equal_nan=True)
    assert np.isclose(
        metrics["false_wet_rainfall_rate"], 1.0, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["missed_wet_rainfall_rate"], 1.0, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_reference"] == 0
    assert metrics["N_nan_estimate"] == 0


def test_calculate_rainfall_metrics_with_nans():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, np.nan, 0, np.nan])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, np.nan, np.nan, np.nan, 1])

    metrics = plg.validation.calculate_rainfall_metrics(ref_array, est_array)

    assert metrics["reference_rainfall_threshold"] == 0.0
    assert metrics["estimates_rainfall_threshold"] == 0.0
    assert np.isclose(
        metrics["pearson_correlation_coefficient"], 0.1186, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["coefficient_of_variation"], 2.2739, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["root_mean_square_error"], 0.6123, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["mean_absolute_error"], 0.3987, atol=0.001, equal_nan=True
    )
    assert np.isclose(metrics["percent_bias"], 46.9194, atol=0.001, equal_nan=True)

    assert np.isclose(
        metrics["reference_mean_rainfall"], 0.263, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_mean_rainfall"], 0.387, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["reference_rainfall_sum"], 2.11, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_rainfall_sum"], 3.10, atol=0.001, equal_nan=True
    )

    assert np.isclose(metrics["false_wet_ratio"], 0.5, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["missed_wet_ratio"], 0.5, atol=0.001, equal_nan=True)
    assert np.isclose(
        metrics["false_wet_rainfall_rate"], 0.55, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["missed_wet_rainfall_rate"], 0.55, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 4
    assert metrics["N_nan_reference"] == 2
    assert metrics["N_nan_estimate"] == 3


def test_calculate_rainfall_metrics_with_zeros():
    ref_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    metrics = plg.validation.calculate_rainfall_metrics(ref_array, est_array)

    assert metrics["reference_rainfall_threshold"] == 0.0
    assert metrics["estimates_rainfall_threshold"] == 0.0
    assert np.isclose(
        metrics["pearson_correlation_coefficient"], np.nan, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["coefficient_of_variation"], np.inf, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["root_mean_square_error"], 0.578, atol=0.001, equal_nan=True
    )
    assert np.isclose(metrics["mean_absolute_error"], 0.350, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["percent_bias"], np.inf, atol=0.001, equal_nan=True)

    assert np.isclose(
        metrics["reference_mean_rainfall"], 0.0, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_mean_rainfall"], 0.350, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["reference_rainfall_sum"], 0.0, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimated_rainfall_sum"], 4.200, atol=0.001, equal_nan=True
    )

    assert np.isclose(metrics["false_wet_ratio"], 0.5, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["missed_wet_ratio"], np.nan, atol=0.001, equal_nan=True)
    assert np.isclose(
        metrics["false_wet_rainfall_rate"], 0.700, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["missed_wet_rainfall_rate"], np.nan, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_reference"] == 0
    assert metrics["N_nan_estimate"] == 0


def test_calculate_wet_dry_metrics_with_nans():
    ref_array = np.array(
        [
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            np.nan,
            np.nan,
            False,
        ]
    )
    est_array = np.array(
        [
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            np.nan,
            np.nan,
            np.nan,
            False,
        ]
    )

    metrics = plg.validation.calculate_wet_dry_metrics(ref_array, est_array)

    assert np.isclose(metrics["matthews_correlation_coefficient"], 0.099, atol=0.01)
    assert np.isclose(metrics["true_wet_ratio"], 0.5, atol=0.01)
    assert np.isclose(metrics["true_dry_ratio"], 0.6, atol=0.01)
    assert np.isclose(metrics["false_wet_ratio"], 0.4, atol=0.01)
    assert np.isclose(metrics["missed_wet_ratio"], 0.5, atol=0.01)
    assert metrics["N_dry_reference"] == 5
    assert metrics["N_wet_reference"] == 4
    assert metrics["N_true_wet"] == 2
    assert metrics["N_true_dry"] == 3
    assert metrics["N_false_wet"] == 2
    assert metrics["N_missed_wet"] == 2
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 3
    assert metrics["N_nan_reference"] == 2
    assert metrics["N_nan_estimate"] == 3


def test_calculate_wet_dry_metrics_without_bools():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, 1, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    metrics = plg.validation.calculate_wet_dry_metrics(ref_array, est_array)

    assert np.isclose(metrics["matthews_correlation_coefficient"], 0.0, atol=0.01)
    assert np.isclose(metrics["true_wet_ratio"], 0.5, atol=0.01)
    assert np.isclose(metrics["true_dry_ratio"], 0.5, atol=0.01)
    assert np.isclose(metrics["false_wet_ratio"], 0.5, atol=0.01)
    assert np.isclose(metrics["missed_wet_ratio"], 0.5, atol=0.01)
    assert metrics["N_dry_reference"] == 6
    assert metrics["N_wet_reference"] == 6
    assert metrics["N_true_wet"] == 3
    assert metrics["N_true_dry"] == 3
    assert metrics["N_false_wet"] == 3
    assert metrics["N_missed_wet"] == 3
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_reference"] == 0
    assert metrics["N_nan_estimate"] == 0


def test_calculate_wet_dry_metrics_replaced_with_zeros():
    ref_array = np.array(
        [True, True, False, False, True, True, False, False, True, True, False, False]
    )
    est_array = np.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    metrics = plg.validation.calculate_wet_dry_metrics(ref_array, est_array)

    assert np.isclose(metrics["matthews_correlation_coefficient"], 0.0, atol=0.01)
    assert np.isclose(metrics["true_wet_ratio"], 0.0, atol=0.01)
    assert np.isclose(metrics["true_dry_ratio"], 1.0, atol=0.01)
    assert np.isclose(metrics["false_wet_ratio"], 0.0, atol=0.01)
    assert np.isclose(metrics["missed_wet_ratio"], 1.0, atol=0.01)
    assert metrics["N_dry_reference"] == 6
    assert metrics["N_wet_reference"] == 6
    assert metrics["N_true_wet"] == 0
    assert metrics["N_true_dry"] == 6
    assert metrics["N_false_wet"] == 0
    assert metrics["N_missed_wet"] == 6
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_reference"] == 0
    assert metrics["N_nan_estimate"] == 0


def test_print_metrics_table(capsys):
    metrics = {
        "reference_rainfall_threshold": 0.1,
        "pearson_correlation_coefficient": 0.85,
        "coefficient_of_variation": 0.1,
        "root_mean_square_error": 1.5,
        "mean_absolute_error": 1.3,
        "percent_bias": 5.0,
        "mean_rainfall_rate_ref": 1.1,
        "false_wet_ratio": 0.2,
        "false_wet_rainfall_rate": 0.5,
        "N_all": 1000,
    }

    plg.validation.print_metrics_table(metrics)

    captured = capsys.readouterr()
    output = captured.out

    assert "Reference rainfall threshold" in output
    assert "Pearson correlation coefficient" in output
    assert "Coefficient of variation" in output
    assert "Root mean square error" in output
    assert "Mean absolute error" in output
    assert "Percent bias" in output
    assert "Mean rainfall rate ref" in output
    assert "False wet ratio" in output
    assert "False wet rainfall rate" in output
    assert "N all" in output


def test_plot_confusion_matrix_count():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    lines, fills = plg.validation.plot_confusion_matrix_count(
        radar_array, cmls_array, ax=ax
    )

    # Check if the return type is correct
    assert isinstance(lines[0], Line2D)
    assert isinstance(fills[0], PolyCollection)
    plt.close("all")


def test_plot_confusion_matrix_count_bin_type():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    lines, fills = plg.validation.plot_confusion_matrix_count(
        radar_array, cmls_array, ax=ax, bin_type="linear"
    )

    # Check if the spacing between the x-values is linear
    assert np.isclose(np.diff(lines[0].get_xdata()).sum(), 98.99, atol=0.01)
    plt.close("all")


def test_plot_confusion_matrix_count_n_sublinks():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise
    threshold = 1

    fig, ax = plt.subplots()
    lines, fills = plg.validation.plot_confusion_matrix_count(
        radar_array,
        cmls_array,
        ref_thresh=threshold,
        est_thresh=threshold,
        ax=ax,
        n_sublinks=50,
    )

    tp_mask = np.logical_and(cmls_array >= threshold, radar_array >= threshold)
    tp, _ = np.histogram(cmls_array[tp_mask], bins=np.linspace(0.01, 100, 101))

    assert lines[0].get_ydata()[0] == tp[0] / 50
    plt.close("all")


def test_plot_confusion_matrix_count_without_ax():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    lines, fills = plg.validation.plot_confusion_matrix_count(radar_array, cmls_array)

    # Check if the return type is correct
    assert isinstance(lines[0], Line2D)
    assert isinstance(fills[0], PolyCollection)
    plt.close("all")


def test_plot_confusion_matrix_sum():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    lines, fills = plg.validation.plot_confusion_matrix_sum(
        radar_array, cmls_array, ax=ax
    )

    # Check if the return type is correct
    assert isinstance(lines[0], Line2D)
    assert isinstance(fills[0], PolyCollection)
    plt.close("all")


# def test_plot_confusion_matrix_count_input_type():
