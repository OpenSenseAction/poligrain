from __future__ import annotations

import numpy as np
import numpy.testing
import xarray as xr

import poligrain as plg


def test_plot_lines():
    ds_cmls = xr.open_dataset("test_data/openMRG_CML_180minutes.nc")
    lines = plg.plot_map.lines(ds_cmls)
    numpy.testing.assert_almost_equal(
        lines.get_paths()[19].vertices,
        np.array([[11.93019, 57.68762], [11.93377, 57.67562]]),
    )
