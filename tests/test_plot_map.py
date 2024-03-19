from __future__ import annotations

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing
import pytest
import xarray as xr

import poligrain as plg


def test_scatter_lines_with_different_args():
    x0 = [1, -1, 0]
    y0 = [3, 3, 2]
    x1 = [0, 1, 0.5]
    y1 = [9, 10, 5]
    c = [0, 1, 3]

    # test with no optional args
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1)
    numpy.testing.assert_almost_equal(
        lines.get_colors(),
        np.array([[0.12156863, 0.46666667, 0.70588235, 1.0]]),
    )
    numpy.testing.assert_almost_equal(
        lines.get_paths()[0].vertices,
        np.array([[1.0, 3.0], [0.0, 9.0]]),
    )
    numpy.testing.assert_almost_equal(
        lines.get_paths()[1].vertices,
        np.array([[-1.0, 3.0], [1.0, 10.0]]),
    )
    assert lines.get_capstyle() == "round"
    assert lines.get_linestyle() == [(0, None)]  # I think None means, a solid line

    # test passing a color
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1, c="r")
    numpy.testing.assert_almost_equal(
        lines.get_colors(),
        np.array([[1.0, 0.0, 0.0, 1.0]]),
    )

    # test line styles
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1, cap_style="butt", line_style=":")
    assert lines.get_capstyle() == "butt"
    numpy.testing.assert_almost_equal(
        np.array(lines.get_linestyle()[0][1]),
        np.array([3.0, 4.949999999999999]),
    )

    # Testing different padding
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1, cap_style="butt", pad_width=1)
    # ...no idea how to do test now if the correct padding was applied...

    # test passing ax
    fig, ax = plt.subplots(figsize=(9, 3))
    _ = plg.plot_map.scatter_lines(x0, y0, x1, y1, c="g", ax=ax)
    assert fig.get_figwidth() == 9

    # test passing a list of colors
    fig, ax = plt.subplots()
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1, c=c, ax=ax)
    with tempfile.NamedTemporaryFile(delete=True) as f:
        fig.savefig(f)
    result = lines.get_colors()
    expected = np.array(
        [
            [0.267004, 0.004874, 0.329415, 1.0],
            [0.190631, 0.407061, 0.556089, 1.0],
            [0.993248, 0.906157, 0.143936, 1.0],
        ],
    )
    numpy.testing.assert_almost_equal(result, expected)

    # test passing a list of colors with different cmap
    fig, ax = plt.subplots()
    lines = plg.plot_map.scatter_lines(x0, y0, x1, y1, c=c, ax=ax, cmap="turbo")
    with tempfile.NamedTemporaryFile(delete=True) as f:
        fig.savefig(f)
    result = lines.get_colors()
    expected = np.array(
        [
            [0.18995, 0.07176, 0.23217, 1.0],
            [0.10342, 0.896, 0.715, 1.0],
            [0.4796, 0.01583, 0.01055, 1.0],
        ],
    )
    numpy.testing.assert_almost_equal(result, expected)

    # test passing a list of colors with different vmin and vmax
    fig, ax = plt.subplots()
    lines = plg.plot_map.scatter_lines(
        x0,
        y0,
        x1,
        y1,
        c=[-1, 0, 50],
        cmap="Grays",
        vmin=0,
        vmax=100,
        ax=ax,
    )
    with tempfile.NamedTemporaryFile(delete=True) as f:
        fig.savefig(f)
    result = lines.get_colors()
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.58608228, 0.58608228, 0.58608228, 1.0],
        ],
    )
    numpy.testing.assert_almost_equal(result, expected)


def test_plot_lines_dataset():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    # evaluate results for plain function and for xarray accessor version
    for lines in [
        plg.plot_map.plot_lines(ds_cmls, line_width=2, line_color="r"),
        ds_cmls.plg.plot_cmls(line_width=2, line_color="r"),
    ]:
        numpy.testing.assert_almost_equal(
            lines.get_paths()[19].vertices,
            np.array([[11.93019, 57.68762], [11.93377, 57.67562]]),
        )


def test_plot_lines_with_dataarray_colored_lines():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    # evaluate results for plain function and for xarray accessor version
    da_rsl = ds_cmls.rsl.isel(sublink_id=0).isel(time=100)
    fig, ax = plt.subplots()
    for lines in [
        plg.plot_map.plot_lines(da_rsl, cmap="YlGnBu", ax=ax),
        da_rsl.plg.plot_cmls(cmap="YlGnBu", ax=ax),
    ]:
        # the `savefig` is required because it seems that `lines.set_array` from within
        # plot_lines only has an effect if the plot is created, e.g. in a notebook
        # or via plt.show(), but we do not want plt.show() because windows have to
        # be closed manually
        with tempfile.NamedTemporaryFile(delete=True) as f:
            fig.savefig(f)
        result = lines.get_colors()[3:5]
        expected = np.array(
            [
                [0.3273664, 0.74060746, 0.75810842, 1.0],
                [0.17296424, 0.62951173, 0.75952326, 1.0],
            ]
        )
        numpy.testing.assert_almost_equal(result, expected)
        ax.cla()


def test_plot_lines_with_dataarray_raise_wrong_shape():
    ds_cmls = xr.open_dataset("tests/test_data/openMRG_CML_180minutes.nc")
    with pytest.raises(ValueError, match="has to be 1D"):
        plg.plot_map.plot_lines(ds_cmls.rsl.isel(sublink_id=0))
