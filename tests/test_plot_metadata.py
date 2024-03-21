from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr

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
