"""Download, cache and load example data.

The functions here work in conjunction with the example data repository
at https://github.com/cchwala/opensense_example_data.

For each new dataset, it must first be added to this data repo and then
the functions here need to be added.

"""

import urllib
from pathlib import Path

import xarray as xr

BASE_URL = "https://github.com/OpenSenseAction/opensense_example_data"
VERSION = "main"


def download_data_file(url, local_path=".", local_file_name=None, print_output=False):
    """Download a file from an URL.

    Parameters
    ----------
    url : str
        The URL directly pointing to the file to be downloaded.
    local_path : str, optional
        The local directory where the fill shall be stored, by default "."
    local_file_name : str or None, optional
        If provided, this is the name of the file locally, by default the
        file name form the URL, i.e. the part after the last `/` will be
        used as file name.
    print_output : bool, optional
        Set to true to get print info on what the function is doing, by default False

    Returns
    -------
    None (if file already exists), or the return message of `urllib.request.urlretrieve`

    """
    if not Path(local_path).exists():
        if print_output:
            print(f"Creating path {local_path}")  # noqa: T201
        Path(local_path).mkdir(parents=True)

    if local_file_name is None:
        local_file_name = url.split("/")[-1]

    if (Path(local_path) / local_file_name).exists():
        print(  # noqa: T201
            f"File already exists at {Path(local_path) / local_file_name}"
        )
        print("Not downloading!")  # noqa: T201
        return None

    if print_output:
        print(f"Downloading {url}")  # noqa: T201
        print(f"to {local_path}/{local_file_name}")  # noqa: T201

    request_return_meassage = urllib.request.urlretrieve(
        url, Path(local_path) / local_file_name
    )
    return request_return_meassage  # noqa: RET504


def load_openmrg(data_dir=".", subset="8d"):
    """Load example OpenMRG data.

    Parameters
    ----------
    data_dir : str, optional
       Directory where the data will be stored. Default is current directory.
    subset : str, optional
       Subset of data to load. Options are '8d' (8 days, native temporal resolution)
       and '5min_2h' (all data aggregated to 5 minute temporal resolution).
       Default is '8d'.

    Returns
    -------
    ds_rad, ds_cmls, ds_gauges_municp, ds_gauge_smhi

    """
    fn = f"openmrg_cml_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenMRG/{fn}"
    data_path = Path(data_dir)
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_cmls = xr.open_dataset(data_path / fn)

    fn = f"openmrg_rad_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_rad = xr.open_dataset(data_path / fn)

    fn = f"openmrg_municp_gauge_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_gauges_municp = xr.open_dataset(data_path / fn)

    fn = f"openmrg_smhi_gauge_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_gauge_smhi = xr.open_dataset(data_path / fn)

    return ds_rad, ds_cmls, ds_gauges_municp, ds_gauge_smhi


def load_openrainer(data_dir=".", subset="8d"):
    """Load OpenRainER example data.

    Parameters
    ----------
    data_dir : str, optional
       Directory where the data will be stored. Default is current directory.
    subset : str, optional
       Subset of data to load. Options are '8d' (8 days, native temporal resolution)
       No other option is currently implemented. Default is '8d'.

    Returns
    -------
    ds_rad, ds_cmls, ds_gauges

    """
    fn = f"openrainer_cml_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenRainER/{fn}"
    data_path = Path(data_dir)
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_cmls = xr.open_dataset(data_path / fn)

    fn = f"openrainer_radar_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenRainER/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_rad = xr.open_dataset(data_path / fn)

    fn = f"openrainer_gauges_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenRainER/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_gauges = xr.open_dataset(data_path / fn)

    return ds_rad, ds_cmls, ds_gauges


def load_ams_pws(data_dir=".", subset="full_dataset"):
    """Load Amsterdam PWS example data.

    Parameters
    ----------
    data_dir : str, optional
       Directory where the data will be stored. Default is current directory.
    subset : str, optional
       Subset of data to load. Options are 'full_dataset' (25 months, native
       temporal resolution) No other option is currently implemented. Default is
       'full_dataset'.

    Returns
    -------
    ds_pws, ds_gauges

    """
    fn = f"AMS_PWS_{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/ams_pws/{fn}"
    data_path = Path(data_dir)
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_pws = xr.open_dataset(data_path / fn)

    fn = f"AMS_point_reference{subset}.nc"
    url = f"{BASE_URL}/raw/{VERSION}/OpenRainER/{fn}"
    download_data_file(url=url, local_file_name=fn, local_path=data_dir)
    ds_gauges = xr.open_dataset(data_path / fn)

    return ds_pws, ds_gauges
