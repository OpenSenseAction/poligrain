from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PolyCollection

import poligrain as plg

def test_plot_hexbin():
    radar_array = np.arange(0, 100, 0.01)
    noise = np.random.normal(loc=0.0, scale=0.1, size=radar_array.shape)
    cmls_array = radar_array + noise

    fig, ax = plt.subplots()

    # Call the function
    hx = plg.validation.plot_hexbin(
        radar_array, 
        cmls_array, 
        ax=ax
    )

    # Check if the return type is correct
    assert isinstance(hx, PolyCollection)


    plt.close("all")