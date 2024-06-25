from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.container import BarContainer

import poligrain as plg


def test_units_frequency():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    assert ds_cmls.frequency.units.lower() == "MHz".lower()


def test_units_length():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    assert ds_cmls.length.units.lower() == "m".lower()


def test_shape_length_frequency_arrays():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    assert (
        ds_cmls.length.broadcast_like(ds_cmls.frequency).shape
        == ds_cmls.frequency.shape
    )


def test_length_values_of_sublinks():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    length = ds_cmls.length.broadcast_like(ds_cmls.frequency)
    assert (
        length.isel(sublink_id=0).to_numpy().all()
        == length.isel(sublink_id=1).to_numpy().all()
    )


def test_plot_len_vs_freq():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    scatter = plg.plot_metadata.plot_len_vs_freq(ds_cmls.length, ds_cmls.frequency)
    len_values = ds_cmls.length.broadcast_like(ds_cmls.frequency).to_numpy() / 1000
    freq_values = ds_cmls.frequency.to_numpy() / 1000
    assert len_values[0][0] == scatter.get_offsets().data[0][0]
    assert freq_values[0][0] == scatter.get_offsets().data[0][1]

    # test passing ax
    fig, ax = plt.subplots(figsize=(5, 4))
    _ = plg.plot_metadata.plot_len_vs_freq(ds_cmls.length, ds_cmls.frequency, ax=ax)
    assert fig.get_figwidth() == 5


def test_plot_len_vs_freq_hexbin_default():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Call the function
    hexbin = plg.plot_metadata.plot_len_vs_freq_hexbin(
        ds_cmls.length, ds_cmls.frequency, ax=ax
    )

    # Check if the return type is correct
    assert isinstance(hexbin, PolyCollection)

    # Check if the labels are set correctly
    assert ax.get_xlabel() == "Length (km)"
    assert ax.get_ylabel() == "Frequency (GHz)"


def test_plot_len_vs_freq_hexbin_no_ax():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")

    # Call the function without an ax
    hexbin = plg.plot_metadata.plot_len_vs_freq_hexbin(
        ds_cmls.length, ds_cmls.frequency
    )

    # Check if the return type is correct
    assert isinstance(hexbin, PolyCollection)


def test_plot_distribution_default():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Call the function to plot length distribution
    hist, bins, patches = plg.plot_metadata.plot_distribution(
        ds_cmls.length, ds_cmls.frequency, variable="length", ax=ax
    )

    # Check if the return types are correct
    assert isinstance(hist, np.ndarray)
    assert isinstance(bins, np.ndarray)
    assert isinstance(patches, BarContainer)
    assert all(isinstance(patch, PathCollection) for patch in patches)

    # Check if the labels are set correctly
    assert ax.get_xlabel() == "Length (km)"
    assert ax.get_ylabel() == "Count (nr. of sublinks)"


def test_plot_distribution_percentage():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Call the function to plot frequency distribution with percentage
    hist, bins, patches = plg.plot_metadata.plot_distribution(
        ds_cmls.length, ds_cmls.frequency, variable="frequency", percentage=True, ax=ax
    )

    # Check if the return types are correct
    assert isinstance(hist, np.ndarray)
    assert isinstance(bins, np.ndarray)
    assert isinstance(patches, BarContainer)
    assert all(isinstance(patch, PathCollection) for patch in patches)


def test_plot_distribution_kwargs():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Call the function with additional kwargs
    kwargs = {"alpha": 0.5}
    hist, bins, patches = plg.plot_metadata.plot_distribution(
        ds_cmls.length, ds_cmls.frequency, variable="length", ax=ax, **kwargs
    )

    # Check if kwargs were applied correctly
    assert patches[0].get_alpha() == 0.5


def test_plot_polarization_default():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Call the function with default parameters
    bars = plg.plot_metadata.plot_polarization(ds_cmls.polarization, ax=ax)

    # Check if the return type is correct
    assert isinstance(bars, BarContainer)

    # Check if the labels are set correctly
    assert ax.get_xlabel() == "Polarization"
    assert ax.get_ylabel() == "Count (nr. of CMLs)"


def test_plot_polarization_count():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()

    # Initialize counts
    count_hh = 0
    count_vv = 0
    count_hv = 0

    for i in range(ds_cmls.polarization.sizes["cml_id"]):
        # Iterate through each cml_id to count the polarization combinations
        sublink_pol = ds_cmls.polarization[:, i].values  # noqa: PD011

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

    # Call the function with example data
    bars = plg.plot_metadata.plot_polarization(ds_cmls.polarization, ax=ax)

    # Check if HH (horizontal), VV (vertical), and HV (mixed) counts are correct
    assert bars[0].get_height() == count_hh
    assert bars[1].get_height() == count_vv
    assert bars[2].get_height() == count_hv

    plt.close("all")
