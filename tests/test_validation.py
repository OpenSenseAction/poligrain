from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.patches import StepPatch

import poligrain as plg


def test_plot_hexbin():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()

    # Call the function
    hx = plg.validation.plot_hexbin(radar_array, cmls_array, ax=ax)

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)

    plt.close("all")


def test_plot_hexbin_without_ax():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    # Call the function without providing an axis
    hx = plg.validation.plot_hexbin(radar_array, cmls_array)

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)

    plt.close("all")


def test_plot_hexbin_without_colorbar():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
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

    assert metrics["ref_thresh"] == 0.1
    assert metrics["est_thresh"] == 0.1
    assert np.isclose(
        metrics["pearson_correlation_coefficient"], -0.387, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["coefficient_of_variation"], 1.767, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["root_mean_square_error"], 0.805, atol=0.001, equal_nan=True
    )
    assert np.isclose(metrics["mean_absolute_error"], 0.678, atol=0.001, equal_nan=True)
    assert np.isclose(metrics["percent_bias"], 2.439, atol=0.001, equal_nan=True)

    assert np.isclose(
        metrics["reference_mean_rainfall"], 0.456, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["estimate_mean_rainfall"], 0.467, atol=0.001, equal_nan=True
    )

    assert np.isclose(
        metrics["false_positive_mean_rainfall"], 1.0, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["false_negative_mean_rainfall"], 1.0, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_ref"] == 0
    assert metrics["N_nan_est"] == 0


def test_calculate_rainfall_metrics_with_nans():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, np.nan, 0, np.nan])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, np.nan, np.nan, np.nan, 1])

    metrics = plg.validation.calculate_rainfall_metrics(ref_array, est_array)

    assert metrics["ref_thresh"] == 0.0
    assert metrics["est_thresh"] == 0.0
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
        metrics["estimate_mean_rainfall"], 0.387, atol=0.001, equal_nan=True
    )

    assert np.isclose(
        metrics["false_positive_mean_rainfall"], 0.55, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["false_negative_mean_rainfall"], 0.55, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 4
    assert metrics["N_nan_ref"] == 2
    assert metrics["N_nan_est"] == 3


def test_calculate_rainfall_metrics_with_zeros():
    ref_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    with pytest.warns(RuntimeWarning):
        metrics = plg.validation.calculate_rainfall_metrics(ref_array, est_array)

    assert metrics["ref_thresh"] == 0.0
    assert metrics["est_thresh"] == 0.0
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
        metrics["estimate_mean_rainfall"], 0.350, atol=0.001, equal_nan=True
    )

    assert np.isclose(
        metrics["false_positive_mean_rainfall"], 0.700, atol=0.001, equal_nan=True
    )
    assert np.isclose(
        metrics["false_negative_mean_rainfall"], np.nan, atol=0.001, equal_nan=True
    )
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_ref"] == 0
    assert metrics["N_nan_est"] == 0


def test_calculate_wet_dry_metrics_with_thresholds():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, 1, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    metrics = plg.validation.calculate_wet_dry_metrics(
        ref_array, est_array, ref_thresh=0.1, est_thresh=0.1
    )

    assert np.isclose(
        metrics["matthews_correlation_coefficient"], -0.169, atol=0.01, equal_nan=True
    )
    assert np.isclose(metrics["true_positive_ratio"], 0.400, atol=0.01, equal_nan=True)
    assert np.isclose(metrics["true_negative_ratio"], 0.428, atol=0.01, equal_nan=True)
    assert np.isclose(metrics["false_positive_ratio"], 0.571, atol=0.01, equal_nan=True)
    assert np.isclose(metrics["false_negative_ratio"], 0.600, atol=0.01, equal_nan=True)
    assert metrics["N_dry_ref"] == 7
    assert metrics["N_wet_ref"] == 5
    assert metrics["N_tp"] == 2
    assert metrics["N_tn"] == 3
    assert metrics["N_fp"] == 4
    assert metrics["N_fn"] == 3
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_ref"] == 0
    assert metrics["N_nan_est"] == 0


def test_calculate_wet_dry_metrics_with_nans():
    ref_array = np.array([1, 1, 0, 0, 0.1, 0.01, 0, 0, 1, np.nan, 0, np.nan])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, np.nan, np.nan, np.nan, 1])

    with pytest.warns(RuntimeWarning):
        metrics = plg.validation.calculate_wet_dry_metrics(ref_array, est_array)

    assert np.isclose(
        metrics["matthews_correlation_coefficient"], np.nan, atol=0.01, equal_nan=True
    )
    assert np.isclose(metrics["true_positive_ratio"], 1.0, atol=0.01, equal_nan=True)
    assert np.isclose(metrics["true_negative_ratio"], np.nan, atol=0.01, equal_nan=True)
    assert np.isclose(
        metrics["false_positive_ratio"], np.nan, atol=0.01, equal_nan=True
    )
    assert np.isclose(metrics["false_negative_ratio"], 0.0, atol=0.01, equal_nan=True)
    assert metrics["N_dry_ref"] == 0
    assert metrics["N_wet_ref"] == 8
    assert metrics["N_tp"] == 8
    assert metrics["N_tn"] == 0
    assert metrics["N_fp"] == 0
    assert metrics["N_fn"] == 0
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 4
    assert metrics["N_nan_ref"] == 2
    assert metrics["N_nan_est"] == 3


def test_calculate_wet_dry_metrics_with_zeros():
    ref_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    est_array = np.array([0, 1, 1, 0, 0, 1, 0.1, 0, 0, 0.1, 1, 0])

    with pytest.warns(RuntimeWarning):
        metrics = plg.validation.calculate_wet_dry_metrics(ref_array, est_array)

    assert np.isclose(
        metrics["matthews_correlation_coefficient"], np.nan, atol=0.01, equal_nan=True
    )
    assert np.isclose(metrics["true_positive_ratio"], 1.0, atol=0.01, equal_nan=True)
    assert np.isclose(metrics["true_negative_ratio"], np.nan, atol=0.01, equal_nan=True)
    assert np.isclose(
        metrics["false_positive_ratio"], np.nan, atol=0.01, equal_nan=True
    )
    assert np.isclose(metrics["false_negative_ratio"], 0.0, atol=0.01, equal_nan=True)
    assert metrics["N_dry_ref"] == 0
    assert metrics["N_wet_ref"] == 12
    assert metrics["N_tp"] == 12
    assert metrics["N_tn"] == 0
    assert metrics["N_fp"] == 0
    assert metrics["N_fn"] == 0
    assert metrics["N_all"] == 12
    assert metrics["N_nan"] == 0
    assert metrics["N_nan_ref"] == 0
    assert metrics["N_nan_est"] == 0


def test_print_metrics_table(capsys):
    metrics = {
        "ref_thresh": 0.1,
        "pearson_correlation_coefficient": 0.85,
        "coefficient_of_variation": 0.1,
        "root_mean_square_error": 1.5,
        "mean_absolute_error": 1.3,
        "percent_bias": 5.0,
        "reference_mean_rainfall": 1.1,
        "false_positive_ratio": 0.2,
        "false_positive_mean_rainfall": 0.5,
        "N_all": 1000,
    }

    print(plg.validation.print_metrics_table(metrics))

    captured = capsys.readouterr()
    output = captured.out

    # assert "Reference rainfall threshold" in output
    assert "Pearson correlation coefficient" in output
    assert "Coefficient of variation" in output
    assert "Root mean square error" in output
    assert "Mean absolute error" in output
    assert "Percent bias" in output
    assert "Mean reference rainfall" in output
    assert "False positive ratio" in output
    assert "False positive mean rainfall" in output
    assert "Total data points" in output


def test_plot_confusion_matrix_count():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_count(radar_array, cmls_array, ax=ax)

    # Check if the return type is correct
    assert isinstance(steps[0], StepPatch)
    plt.close("all")


def test_plot_confusion_matrix_count_bin_type():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_count(
        radar_array, cmls_array, ax=ax, bin_type="log"
    )

    # Check if the spacing between the x-values is linear
    assert np.isclose(np.diff(steps[0].get_data().edges).sum(), 99.90, atol=0.01)
    plt.close("all")


def test_plot_confusion_matrix_count_custom_bins():
    radar_array = np.arange(0, 200, 0.1)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_count(
        radar_array, cmls_array, ax=ax, bin_type="log", bins=np.linspace(0, 200, 201)
    )

    # Check if the spacing of the custom bins supplied is used in the function
    assert np.isclose(np.diff(steps[0].get_data().edges).sum(), 200, atol=0.1)
    plt.close("all")


def test_plot_confusion_matrix_count_y_normalized():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise
    threshold = 1

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_count(
        radar_array,
        cmls_array,
        ref_thresh=threshold,
        est_thresh=threshold,
        ax=ax,
        normalize_y=50,
    )

    tp_mask = np.logical_and(cmls_array >= threshold, radar_array >= threshold)
    tp, _ = np.histogram(cmls_array[tp_mask], bins=np.linspace(0.1, 100, 101))

    # Check if the height of the first step corresponds to the input histogram
    assert steps[0].get_data()[0][0] == tp[0] / 50
    plt.close("all")


def test_plot_confusion_matrix_count_without_ax():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    steps = plg.validation.plot_confusion_matrix_count(radar_array, cmls_array)

    # Check if the return type is correct
    assert isinstance(steps[0], StepPatch)
    plt.close("all")


def test_plot_confusion_matrix_sum():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_sum(
        radar_array, cmls_array, time_interval=5, ax=ax
    )

    # Check if the return type is correct
    assert isinstance(steps[0], StepPatch)
    plt.close("all")


def test_plot_confusion_matrix_sum_bin_type():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_sum(
        radar_array, cmls_array, time_interval=5, ax=ax, bin_type="log"
    )

    # Check if the spacing between the x-values is linear
    assert np.isclose(np.diff(steps[0].get_data().edges).sum(), 99.90, atol=0.01)
    plt.close("all")


def test_plot_confusion_matrix_sum_custom_bins():
    radar_array = np.arange(0, 200, 0.1)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_sum(
        radar_array,
        cmls_array,
        ax=ax,
        bin_type="log",
        time_interval=5,
        bins=np.linspace(0, 200, 201),
    )

    # Check if the spacing of the custom bins supplied is used in the function
    assert np.isclose(np.diff(steps[0].get_data().edges).sum(), 200, atol=0.1)
    plt.close("all")


def test_plot_confusion_matrix_sum_unsupported_bins():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    with pytest.raises(
        ValueError, match="Unsupported bin_type, must be 'linear' or 'log'."
    ):
        plg.validation.plot_confusion_matrix_sum(
            radar_array, cmls_array, time_interval=5, bin_type="unsupported"
        )


def test_plot_confusion_matrix_sum_y_normalized():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise
    threshold = 0

    fig, ax = plt.subplots()
    steps = plg.validation.plot_confusion_matrix_sum(
        radar_array,
        cmls_array,
        ref_thresh=threshold,
        est_thresh=threshold,
        time_interval=5,
        ax=ax,
        normalize_y=50,
    )

    tp_mask = np.logical_and(cmls_array >= threshold, radar_array >= threshold)
    tp, def_bins = np.histogram(cmls_array[tp_mask], bins=np.linspace(0.1, 100, 101))

    bin_cent = (def_bins[:-1] + def_bins[1:]) / 2
    rate_to_sum = 5 / 60

    # Check if the height of the first step corresponds to the input histogram
    assert steps[0].get_data()[0][-1] == (bin_cent[-1] * rate_to_sum) * (tp[-1] / 50)
    plt.close("all")


def test_plot_confusion_matrix_sum_without_ax():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.default_rng().normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    steps = plg.validation.plot_confusion_matrix_sum(
        radar_array, cmls_array, time_interval=15
    )

    # Check if the return type is correct
    assert isinstance(steps[0], StepPatch)
    plt.close("all")
