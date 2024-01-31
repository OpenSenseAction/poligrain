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
    return ds_points.x, ds_points.y


def project_point_coordinates(
    x: xr.DataArray,
    y: xr.DataArray,
    target_projection: str,
    source_projection: str = "EPSG:4326",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Project coordinates x and y of point data

    Note that `x` and `y` have to be `xarray.DataArray` so that we can return
    the projected coordinates also as `xarray.DataArray` with the correct
    `coord` data so that they can easily and safely added to an existing
    `xarray.Dataset`, e.g. like the following code:

    >>> ds_gauges.coords["x"], ds_gauges.coords["y"] = plg.spatial.project_point_coordinates(
    ...     ds_gauges.lon, ds_gauges.lat, target_projection="EPSG:25832",
    ...     )

    Parameters
    ----------
    x : xr.DataArray
        The coordinates along the x-axis
    y : xr.DataArray
        The coordinates along the y-axis
    target_projection : str
        An EPSG string that defines the projection the points shall be projected too,
        e.g. "EPSG:25832" for UTM zone 32N
    source_projection : str, optional
        An EPSG string that defines the projection of the supplied `x` and `y` data,
        by default "EPSG:4326"

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The projected coordinates
    """
    transformer = pyproj.Transformer.from_crs(
        crs_to=target_projection, crs_from=source_projection, always_xy=True
    )
    x_projected, y_projected = transformer.transform(x, y)  # pylint: disable=unpacking-non-sequence

    x_projected = xr.DataArray(data=x_projected, coords=x.coords, name="x")
    y_projected = xr.DataArray(data=y_projected, coords=y.coords, name="y")
    x_projected.attrs["projection"] = target_projection
    y_projected.attrs["projection"] = target_projection
    return x_projected, y_projected


def calc_point_to_point_distances(
    ds_points_a: xr.DataArray | xr.Dataset, ds_points_b: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Calculate the distance between the point coordinates of two datasets.

    Parameters
    ----------
    ds_points_a : xr.DataArray | xr.Dataset
        _description_
    ds_points_b : xr.DataArray | xr.Dataset
        _description_

    Returns
    -------
    xr.DataArray
        Distance matrix in meters, assuming `x` and `y` coordinate variables in the
        supplied data are projected to something like UTM. The dimensions of the matrix
        are the `id` dimensions of the two input datasets. The `id` values are also
        provided along each dimension. The second dimension name is appended with `_neighbor`.
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
