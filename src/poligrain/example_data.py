"""Download, cache and load example data."""

import urllib
from pathlib import Path

import xarray as xr

_default_cache_dir_name = "opensense_example_data"
base_url = "https://github.com/cchwala/opensense_example_data"
version = "main"


def download_data_file(url, local_path=".", local_file_name=None, print_output=False):
    """Download a file from an URL.

    Parameters
    ----------
    url : _type_
        _description_
    local_path : str, optional
        _description_, by default "."
    local_file_name : _type_, optional
        _description_, by default None
    print_output : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if not Path(local_path).exists():
        if print_output:
            print(f"Creating path {local_path}")  # noqa: T201
        Path.makedirs(local_path)

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


def load_openmrg_5min_2h():
    """Load 2.5 hours of OpenMRG data with a 5-min resolution.

    Returns
    -------
    ds_rad, ds_cmls, ds_gauges_municp, ds_gauge_smhi
    """
    fn = "openmrg_cml_5min_2h.nc"
    url = f"{base_url}/raw/{version}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn)
    ds_cmls = xr.open_dataset(fn)

    fn = "openmrg_rad_5min_2h.nc"
    url = f"{base_url}/raw/{version}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn)
    ds_rad = xr.open_dataset(fn)

    fn = "openmrg_municp_gauge_5min_2h.nc"
    url = f"{base_url}/raw/{version}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn)
    ds_gauges_municp = xr.open_dataset(fn)

    fn = "openmrg_smhi_gauge_5min_2h.nc"
    url = f"{base_url}/raw/{version}/OpenMRG/{fn}"
    download_data_file(url=url, local_file_name=fn)
    ds_gauge_smhi = xr.open_dataset(fn)

    return ds_rad, ds_cmls, ds_gauges_municp, ds_gauge_smhi
