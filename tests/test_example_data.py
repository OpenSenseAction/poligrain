from pathlib import Path

import numpy as np
import numpy.testing as npt
import xarray as xr

import poligrain as plg


def test_download_data_file():
    BASE_URL = "https://github.com/cchwala/opensense_example_data"
    VERSION = "main"

    fn = "openmrg_municp_gauge_5min_2h.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenMRG/{fn}"
    data_dir = "example_data"
    plg.example_data.download_data_file(
        url=url, local_file_name=fn, local_path=data_dir
    )
    ds_gauges = xr.open_dataset(Path(data_dir) / fn)

    npt.assert_almost_equal(
        ds_gauges.lon.data[:3], np.array([11.943145, 12.035572, 12.073303])
    )
    npt.assert_almost_equal(
        ds_gauges.station_id.data[:3],
        np.array(
            [
                0,
                1,
                2,
            ]
        ),
    )
