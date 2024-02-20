from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing
import pytest
import xarray as xr

import poligrain as plg


def test_plot_lines_dataset():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    lines = plg.plot_map.plot_lines(ds_cmls)
    numpy.testing.assert_almost_equal(
        lines.get_paths()[19].vertices,
        np.array([[11.93019, 57.68762], [11.93377, 57.67562]]),
    )


def test_plot_lines_with_dataarray_colored_lines():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    fig, ax = plt.subplots()
    lines = plg.plot_map.plot_lines(
        ds_cmls.rsl.isel(sublink_id=0).isel(time=100),
        cmap="YlGnBu",
    )
    # the `savefig` is required because it seems that `lines.set_array` from within
    # plot_lines only has an effect if the plot is created, e.g. in a notebook
    # or via plt.show(), but we do not want plt.show() because windows have to
    # be closed manually
    plt.savefig("foo.png")
    result = lines.get_colors()[3:5]
    expected = np.array(
        [
            [0.3273664, 0.74060746, 0.75810842, 1.0],
            [0.17296424, 0.62951173, 0.75952326, 1.0],
        ]
    )
    numpy.testing.assert_almost_equal(result, expected)


def test_plot_lines_with_dataarray_raise_wrong_shape():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    with pytest.raises(ValueError, match="has to be 1D"):
        plg.plot_map.plot_lines(ds_cmls.rsl.isel(sublink_id=0))
