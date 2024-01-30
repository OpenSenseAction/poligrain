"""Functions for calculating spatial distance, intersections and finding neighbors"""

import pyproj
import scipy
import xarray as xr


def get_point_xy(
    ds_points: xr.DataArray | xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Get x and y coordinate data for point Dataset or DataArray

    Use this function instead of just getting the x and y variables from the
    xarray.Dataset or DataArray, because it will do some additional checks.
    Furthermore it will facility adapting to changing naming conventions
    in the future.

    Parameters
    ----------
    ds_points : xr.DataArray | xr.Dataset
        The Dataset or DataArray to get x and y from. It has to obey to the
        OPENSENSE data format conventions.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        x and y as xr.DataArray
    """
    assert len(ds_points.x.dims) == 1
    assert len(ds_points.y.dims) == 1
    assert ds_points.x.dims == ds_points.y.dims
    return ds_points.x, ds_points


def project_point_coordinates(
    x: xr.DataArray,
    y: xr.DataArray,
    target_projection: str,
    source_projection: str = "EPSG:4326",
) -> tuple[xr.DataArray, xr.DataArray]:
    """_summary_

    Parameters
    ----------
    x : xr.DataArray
        _description_
    y : xr.DataArray
        _description_
    target_projection : str
        _description_
    source_projection : _type_, optional
        _description_, by default "EPSG:4326"

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        _description_
    """
    transformer = pyproj.Transformer.from_crs(
        crs_to=target_projection, crs_from=source_projection, always_xy=True
    )
    x_projected, y_projected = transformer.transform(x, y)  # pylint: disable=unpacking-non-sequence

    x_projected = xr.DataArray(data=x_projected, dims=x.dims, name="x")
    y_projected = xr.DataArray(data=y_projected, dims=y.dims, name="y")
    x_projected.attrs["projection"] = target_projection
    y_projected.attrs["projection"] = target_projection
    return x_projected, y_projected


def calc_point_to_point_distances(
    ds_points_a: xr.DataArray | xr.Dataset, ds_points_b: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Calculate the distance between two datasets of points

    Returns
    -------
    _type_
        _description_
    """
    x_a, y_a = get_point_xy(ds_points_a)
    x_b, y_b = get_point_xy(ds_points_b)

    distance_matrix = scipy.spatial.distance_matrix(
        x=list(zip(x_a.values, y_a.values, strict=True)),
        y=list(zip(x_b.values, y_b.values, strict=True)),
    )

    dim_a = x_a.dims[0]
    dim_b = x_b.dims[0] + "_neighbor"
    return xr.DataArray(
        data=distance_matrix,
        dims=(dim_a, dim_b),
        coords={
            dim_a: (dim_a, x_a[x_a.dims[0]].data),
            dim_b: (dim_b, x_b[x_b.dims[0]].data),
        },
    )
