from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

import poligrain as plg
from __future__ import annotations


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
    import matplotlib.pyplot as plt

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


def test_calculate_rainfall_metrics():

    # check if statements

    # check the working of the metrics with sample data set with known metrics

def test_calculate_wet_dry_metrics():

    # check if statements

    # check the working of the metrics with sample data set with known metrics
    

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










