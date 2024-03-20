# %%
from __future__ import annotations

import xarray as xr


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
        length.isel(sublink_id=0).values.all() == length.isel(sublink_id=1).values.all()  # noqa: PD011
    )
