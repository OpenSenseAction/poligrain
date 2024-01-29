"""Functions for calculating spatial distance, intersections and finding neighbors"""

import pyproj
import scipy
import xarray as xr


def get_point_xy(ds_points: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
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
    transformer = pyproj.Transformer.from_crs(
        crs_to=target_projection, crs_from=source_projection, always_xy=True
    )
    x_projected, y_projected = transformer.transform(x, y)

    x_projected = xr.DataArray(data=x_projected, dims=x.dims, name="x")
    y_projected = xr.DataArray(data=y_projected, dims=y.dims, name="y")
    x_projected.attrs["projection"] = target_projection
    y_projected.attrs["projection"] = target_projection
    return x_projected, y_projected


def calc_point_to_point_distances(
    ds_points_a: xr.DataArray, ds_points_b: xr.DataArray
) -> xr.DataArray:
    """Calculate the distance between two datasets of points"""

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
            dim_a: (dim_a, x_a[x_a.dims[0]].to_numpy()),
            dim_b: (dim_b, x_b[x_b.dims[0]].to_numpy()),
        },
    )
