import tempfile
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

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        data_dir_in_tmp_dir = Path(tmp_dir_name) / data_dir

        # test download with specific local file name
        plg.example_data.download_data_file(
            url=url,
            local_file_name="some_file_name.nc",
            local_path=data_dir_in_tmp_dir,
            print_output=True,
        )
        ds_gauges = xr.open_dataset(Path(data_dir_in_tmp_dir) / "some_file_name.nc")

        npt.assert_almost_equal(
            ds_gauges.lon.data[:3], np.array([11.943145, 12.035572, 12.073303])
        )
        npt.assert_almost_equal(ds_gauges.station_id.data[:3], np.array([0, 1, 2]))

        # call download without specifying local file name and check that
        # file gets correct filename which is the last part of the URL
        # which is `fn` since we build it like that above.
        return_code = plg.example_data.download_data_file(
            url=url,
        )
        assert Path(fn).exists()

        # download again to the same local target, which should not do anything and
        # then return None
        return_code = plg.example_data.download_data_file(
            url=url,
        )
        assert return_code is None

        # close file, because otherwise CI on Windows fails when trying to delete dir
        ds_gauges.close()


def test_load_openmrg_5min_2h():
    # We download the data and just check a little bit of the data. Here
    # we do not yet check that data format conventions are correct.
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        (
            ds_rad,
            ds_cmls,
            ds_gauges_municp,
            ds_gauge_smhi,
        ) = plg.example_data.load_openmrg_5min_2h(data_dir=tmp_dir_name)

        # Check radar data
        npt.assert_almost_equal(
            ds_rad.rainfall_amount.data[10, 20:22, 15:18],
            np.array(
                [
                    [0.03038611, 0.02556672, 0.02556672],
                    [0.04546478, 0.03409377, 0.04292152],
                ]
            ),
        )

        # Check CML data
        npt.assert_almost_equal(
            ds_cmls.isel(cml_id=42).R.data[11:14],
            np.array([0.58149174, 0.43223663, 0.33730422]),
        )

        # Check content of muncip gauge data
        npt.assert_almost_equal(
            ds_gauges_municp.lon.data[:3], np.array([11.943145, 12.035572, 12.073303])
        )
        npt.assert_almost_equal(
            ds_gauges_municp.station_id.data[:3], np.array([0, 1, 2])
        )

        # Check data from SMHI gauge
        npt.assert_almost_equal(
            ds_gauge_smhi.rainfall_amount.isel(station_id=0).data[11:14],
            np.array([0.63333333, 0.63333333, 0.53333333]),
        )
