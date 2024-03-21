"""Functions for calculating spatial distance, intersections and finding neighbors."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pyproj
import scipy
import sparse
import xarray as xr
from shapely.geometry import LineString, Polygon


def get_point_xy(
    ds_points: xr.DataArray | xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Get x and y coordinate data for point Dataset or DataArray.

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
    """Project coordinates x and y of point data.

    Note that `x` and `y` have to be `xarray.DataArray` so that we can return
    the projected coordinates also as `xarray.DataArray` with the correct
    `coord` data so that they can easily and safely added to an existing
    `xarray.Dataset`, e.g. like the following code:

    >>> ds.coords["x"], ds.coords["y"] = plg.spatial.project_point_coordinates(
    ...     ds.lon, ds.lat, target_projection="EPSG:25832",
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
        supplied data are projected to something like UTM. The dimensions of the
        matrix are the `id` dimensions of the two input datasets. The `id` values
        are also provided along each dimension. The second dimension name is appended
        with `_neighbor`.
    """
    x_a, y_a = get_point_xy(ds_points_a)
    x_b, y_b = get_point_xy(ds_points_b)

    distance_matrix = scipy.spatial.distance_matrix(
        x=list(zip(x_a.values, y_a.values)),
        y=list(zip(x_b.values, y_b.values)),
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


class GridAtLines():
    def __init__(
            self,
            da_gridded_data,
            ds_line_data,
            grid_point_location='center',
            sensor_id=None,
    ):
        if sensor_id is None:
            try:
                sensor_id = ds_line_data.cml_id
            except AttributeError:
                sensor_id = ds_line_data.sml_id

        self.intersect_weights = calc_sparse_intersect_weights_for_several_cmls(
            x1_line=ds_line_data.site_0_lon.values,
            y1_line=ds_line_data.site_0_lat.values,
            x2_line=ds_line_data.site_1_lon.values,
            y2_line=ds_line_data.site_1_lat.values,
            cml_id=ds_line_data.cml_id,
            x_grid=da_gridded_data.lon.values,
            y_grid=da_gridded_data.lat.values,
            grid_point_location=grid_point_location,
        )

    def __call__(self, da_gridded_data):
        gridded_data_along_line = get_grid_time_series_at_intersections(
            grid_data=da_gridded_data,
            intersect_weights=self.intersect_weights,
        )

        gridded_data_along_line['time'] = da_gridded_data.time
        gridded_data_along_line['site_0_lon'] = self.intersect_weights.site_0_lon
        gridded_data_along_line['site_1_lat'] = self.intersect_weights.site_1_lat
        gridded_data_along_line['site_1_lon'] = self.intersect_weights.site_1_lon
        gridded_data_along_line['site_0_lat'] = self.intersect_weights.site_0_lat

        return gridded_data_along_line


class GridAtPoints():
    def __init__(self, da_gridded_data, da_point_data, nnear=9, stat='best'):
        # TODO: account for `grid_point_location` which is not yet done here.

        # TODO: Copy-paste the relevant stuff from wradlib and add to own code
        import wradlib

        # Get radar pixel coordinates as (N, 2) array
        x_grid = da_gridded_data.lon.to_numpy()
        y_grid = da_gridded_data.lat.to_numpy()
        xy_grid = np.array(list(zip(x_grid.flatten(), y_grid.flatten(), strict=True)))

        # Initialize function to get grid values at points
        xy_points = np.stack([da_point_data.lon, da_point_data.lat], axis=1)
        self.raw_at_obs_adjuster = wradlib.adjust.RawAtObs(
            obs_coords=xy_points, #df_stations.loc[:, ["x", "y"]].values,
            raw_coords=xy_grid,
            nnear=nnear,
            stat=stat,
        )

    def __call__(self, da_gridded_data, da_point_data):
        gridded_data_at_point_list = []
        for t in da_gridded_data.time.to_numpy():
            da_gridded_data_t = da_gridded_data.sel(time=t)
            da_point_data_t = da_point_data.sel(time=t)

            gridded_data_at_point_list.append(
                self.raw_at_obs_adjuster(
                    obs=da_point_data_t.to_numpy(),
                    raw=da_gridded_data_t.to_numpy().flatten(),
                )
            )

        return xr.DataArray(
            data=gridded_data_at_point_list,
            dims=da_point_data.dims,
            coords={
                'time': da_gridded_data.time,
                'lon':  da_point_data.lon,
                'lat': da_point_data.lat,
            }
        )


def get_grid_time_series_at_intersections(grid_data, intersect_weights):
    """Get time series from gird data using sparse intersection weights.

    Time series of grid data are derived via intersection weights of CMLs.
    Please note that it is crucial to have the correct order of dimensions, see
    parameter list below.

    Input can be ndarrays or xarray.DataArrays. If at least one input is a
    DataArray, a DataArray is returned.

    Parameters
    ----------
    grid_data: ndarray or xarray.DataArray
        3-D data of the gridded data we want to extract time series from at the
        given pixel intersection. The order of dimensions must be ('time', 'y', 'x').
        The size in the `x` and `y` dimension must be the same as in the intersection
        weights.
    intersect_weights: ndarray or xarray.DataArray
        3-D data of intersection weights. The order of dimensions must be
        ('cml_id', 'y', 'x'). The size in the `x` and `y` dimension must be the
        same as in the grid data. Intersection weights do not have to be a
        `sparse.array` but will be converted to one internally before doing a
        `sparse.tensordot` contraction.

    Returns
    -------
    grid_intersect_timeseries: ndarray or xarray.DataArray
        The time series for each grid intersection. If at least one of the inputs is
        a xarray.DataArray, a xarray.DataArray is returned. Coordinates are
        derived from the input.
    DataArrays.

    """
    return_a_dataarray = False
    try:
        intersect_weights_coords = intersect_weights.coords
        # from here on we only want to deal with the actual array
        intersect_weights = intersect_weights.data
        return_a_dataarray = True
    except AttributeError:
        pass
    try:
        grid_data_coords = grid_data.coords
        # from here on we only want to deal with the actual array
        grid_data = grid_data.data
        return_a_dataarray = True
    except AttributeError:
        pass

    # Assure that we use a sparse matrix for the weights, because, besides
    # being much faster for large tensordot computation, it can deal with
    # NaN better. If the weights are passed to `sparse.tensordot` as numpy
    # arrays, the value for each time series for a certain point in time is NaN
    # if there is at least one nan in the grid at that point in time. We only
    # want NaN in the time series if the intersection intersects with a NaN grid pixel.
    intersect_weights = sparse.asCOO(intersect_weights, check=False)

    grid_intersect_timeseries = sparse.tensordot(
        grid_data,
        intersect_weights,
        axes=[[1, 2], [1, 2]],
    )

    if return_a_dataarray:
        coords = {}
        try:
            dim_0_name = grid_data_coords.dims[0]
            dim_0_values = grid_data_coords[dim_0_name].values
        except NameError:
            dim_0_name = "time"
            dim_0_values = np.arange(grid_intersect_timeseries.shape[0])
        try:
            dim_1_name = intersect_weights_coords.dims[0]
            dim_1_values = intersect_weights_coords[dim_1_name].values
        except NameError:
            dim_1_name = "cml_id"
            dim_1_values = np.arange(grid_intersect_timeseries.shape[1])
        grid_intersect_timeseries = xr.DataArray(
            data=grid_intersect_timeseries,
            dims=(dim_0_name, dim_1_name),
            coords={dim_0_name: dim_0_values, dim_1_name: dim_1_values},
        )

    return grid_intersect_timeseries


def calc_sparse_intersect_weights_for_several_cmls(
    x1_line,
    y1_line,
    x2_line,
    y2_line,
    cml_id,
    x_grid,
    y_grid,
    grid_point_location="center",
):
    """Calculate sparse intersection weights matrix for several CMLs.

    This function just loops over `calc_intersect_weights` for several CMLs, but
    stores the intersection weight matrices as sparase matrix to save space and
    to allow faster calculation with `sparse.tensordot` afterwards.

    Function arguments are the same as in `calc_intersect_weights`, except that
    we take a 1D array or list of line coordinates here.

    Parameters
    ----------
    x1_line : 1D-array or list of float
    y1_line : 1D-array or list of float
    x2_line : 1D-array or list of float
    y2_line : 1D-array or list of float
    cml_id: 1D-array or list of strings
    x_grid : 2D array
        x-coordinates of grid points
    y_grid : 2D array
        y-coordinates of grid points
    grid_point_location : str, optional
        The only option currently is `center` which assumes that the
        coordinates in `xr_ds` represent the centers of grid cells

    Returns
    -------
    intersect : xarray.DataArray with sparse intersection weights
        The variables `x_grid` and `y_grid` are used as coordinates.
    """
    intersect_weights_list = []
    for i in range(len(cml_id)):
        intersect_weights = calc_intersect_weights(
            x1_line=x1_line[i],
            x2_line=x2_line[i],
            y1_line=y1_line[i],
            y2_line=y2_line[i],
            x_grid=x_grid,
            y_grid=y_grid,
            grid_point_location=grid_point_location,
        )
        intersect_weights_list.append(sparse.COO.from_numpy(intersect_weights))

    da_intersect_weights = xr.DataArray(
        data=sparse.stack(intersect_weights_list),
        dims=("cml_id", "y", "x"),
        coords={
            "x_grid": (("y", "x"), x_grid),
            "y_grid": (("y", "x"), y_grid),
        },
    )
    da_intersect_weights.coords["cml_id"] = cml_id

    return da_intersect_weights


def calc_intersect_weights(
    x1_line,
    y1_line,
    x2_line,
    y2_line,
    x_grid,
    y_grid,
    grid_point_location="center",
    offset=None,
):
    """Calculate intersecting weights for a line and a grid.

    Calculate the intersecting weights for the line defined by `x1_line`,
    `y1_line`, `x2_line` and `y2_line` and the grid defined by the x- and y-
    grid points from `x_grid` and `y_grid`.

    Parameters
    ----------
    x1_line : float
    y1_line : float
    x2_line : float
    y2_line : float
    x_grid : 2D array
        x-coordinates of grid points
    y_grid : 2D array
        y-coordinates of grid points
    grid_point_location : str, optional
        The only option currently is `center` which assumes that the
        coordinates in `xr_ds` represent the centers of grid cells
    offset : float, optional
        The offset in units of the coordinates to constrain the calculation
        of intersection to a bounding box around the CML coordinates. The
        offset specifies by how much this bounding box will be larger then
        the width- and height-extent of the CML coordinates.

    Returns
    -------
    intersect : array
        2D array of intersection weights with shape of the longitudes- and
        latitudes grid of `xr_ds`
    pixel_poly_list : list, optional
        List of shapely.Polygons which were used to calculate intersections

    """
    x_grid = x_grid.astype("float64")
    y_grid = y_grid.astype("float64")

    grid = np.stack([x_grid, y_grid], axis=2)

    # Convert CML path to shapely line
    link = LineString([(x1_line, y1_line), (x2_line, y2_line)])

    # Derive grid cell width to set bounding box offset
    ll_cell = grid[0, 1, 0] - grid[0, 0, 0]
    ul_cell = grid[-1, 1, 0] - grid[-1, 0, 0]
    lr_cell = grid[0, -1, 0] - grid[0, -2, 0]
    ur_cell = grid[-1, -1, 0] - grid[-1, -2, 0]
    offset_calc = max(ll_cell, ul_cell, lr_cell, ur_cell)

    # Set bounding box offset
    if offset is None:
        offset = offset_calc

    # Set bounding box
    x_max = max([x1_line, x2_line])
    x_min = min([x1_line, x2_line])
    y_max = max([y1_line, y2_line])
    y_min = min([y1_line, y2_line])
    bounding_box = ((x_grid > x_min - offset) & (x_grid < x_max + offset)) & (
        (y_grid > y_min - offset) & (y_grid < y_max + offset)
    )

    # Calculate polygon corners assuming that `grid` defines the center
    # of each grid cell
    if grid_point_location == "center":
        grid_corners = _calc_grid_corners_for_center_location(grid)
    elif grid_point_location == "lower_left":
        grid_corners = _calc_grid_corners_for_lower_left_location(grid)
    else:
        raise ValueError(
            "`grid_point_location` = %s not implemented" % grid_point_location  # pylint: disable=C0209
        )

    # Find intersection
    intersect = np.zeros([grid.shape[0], grid.shape[1]])
    pixel_poly_list = []
    # Iterate only over the indices within the bounding box and
    # calculate the intersect weigh for each pixel
    ix_in_bbox = np.where(bounding_box == True)  # noqa: E712 # pylint: disable=C0121
    for i, j in zip(ix_in_bbox[0], ix_in_bbox[1]):
        pixel_poly = Polygon(
            [
                grid_corners.ll_grid[i, j],
                grid_corners.lr_grid[i, j],
                grid_corners.ur_grid[i, j],
                grid_corners.ul_grid[i, j],
            ]
        )
        pixel_poly_list.append(pixel_poly)

        c = link.intersection(pixel_poly)
        if not c.is_empty:
            intersect[i][j] = c.length / link.length

    return intersect


def _calc_grid_corners_for_center_location(grid):
    """Parameters.

    ----------
    grid : array
        3D matrix holding x and y grids. Shape of `grid` must be
        (height, width, 2).

    Returns
    -------
    namedtuple with the grids for the four corners of the grid defined
    by points at the lower left corner

    """
    grid = grid.astype("float64")

    # Upper right
    ur_grid = np.zeros_like(grid)
    ur_grid[0:-1, 0:-1, :] = (grid[0:-1, 0:-1, :] + grid[1:, 1:, :]) / 2.0
    ur_grid[-1, :, :] = ur_grid[-2, :, :] + (ur_grid[-2, :, :] - ur_grid[-3, :, :])
    ur_grid[:, -1, :] = ur_grid[:, -2, :] + (ur_grid[:, -2, :] - ur_grid[:, -3, :])
    # Upper left
    ul_grid = np.zeros_like(grid)
    ul_grid[0:-1, 1:, :] = (grid[0:-1, 1:, :] + grid[1:, :-1, :]) / 2.0
    ul_grid[-1, :, :] = ul_grid[-2, :, :] + (ul_grid[-2, :, :] - ul_grid[-3, :, :])
    ul_grid[:, 0, :] = ul_grid[:, 1, :] - (ul_grid[:, 2, :] - ul_grid[:, 1, :])
    # Lower right
    lr_grid = np.zeros_like(grid)
    lr_grid[1:, 0:-1, :] = (grid[1:, 0:-1, :] + grid[:-1, 1:, :]) / 2.0
    lr_grid[0, :, :] = lr_grid[1, :, :] - (lr_grid[2, :, :] - lr_grid[1, :, :])
    lr_grid[:, -1, :] = lr_grid[:, -2, :] + (lr_grid[:, -2, :] - lr_grid[:, -3, :])
    # Lower left
    ll_grid = np.zeros_like(grid)
    ll_grid[1:, 1:, :] = (grid[1:, 1:, :] + grid[:-1, :-1, :]) / 2.0
    ll_grid[0, :, :] = ll_grid[1, :, :] - (ll_grid[2, :, :] - ll_grid[1, :, :])
    ll_grid[:, 0, :] = ll_grid[:, 1, :] - (ll_grid[:, 2, :] - ll_grid[:, 1, :])

    GridCorners = namedtuple(
        "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
    )

    return GridCorners(
        ur_grid=ur_grid, ul_grid=ul_grid, lr_grid=lr_grid, ll_grid=ll_grid
    )


def _calc_grid_corners_for_lower_left_location(grid):
    """Parameters.

    ----------
    grid : array
        3D matrix holding x and y grids. Shape of `grid` must be
        (height, width, 2).

    Returns
    -------
    namedtuple with the grids for the four corners around the
    central grid points

    """
    grid = grid.astype("float64")

    if (np.diff(grid[:, :, 0], axis=1) < 0).any():
        raise ValueError("x values must be ascending along axis 1")  # noqa: EM101
    if (np.diff(grid[:, :, 1], axis=0) < 0).any():
        raise ValueError("y values must be ascending along axis 0")  # noqa: EM101

    # Upper right
    ur_grid = np.zeros_like(grid)
    ur_grid[0:-1, 0:-1, :] = grid[1:, 1:, :]
    ur_grid[-1, :, :] = ur_grid[-2, :, :] + (ur_grid[-2, :, :] - ur_grid[-3, :, :])
    ur_grid[:, -1, :] = ur_grid[:, -2, :] + (ur_grid[:, -2, :] - ur_grid[:, -3, :])
    # Upper left
    ul_grid = np.zeros_like(grid)
    ul_grid[0:-1, 0:-1, :] = grid[1:, 0:-1, :]
    ul_grid[-1, :, :] = ul_grid[-2, :, :] + (ul_grid[-2, :, :] - ul_grid[-3, :, :])
    ul_grid[:, -1, :] = ul_grid[:, -2, :] + (ul_grid[:, -2, :] - ul_grid[:, -3, :])
    # Lower right
    lr_grid = np.zeros_like(grid)
    lr_grid[0:-1, 0:-1, :] = grid[0:-1, 1:, :]
    lr_grid[-1, :, :] = lr_grid[-2, :, :] + (lr_grid[-2, :, :] - lr_grid[-3, :, :])
    lr_grid[:, -1, :] = lr_grid[:, -2, :] + (lr_grid[:, -2, :] - lr_grid[:, -3, :])
    # Lower left
    ll_grid = grid.copy()

    GridCorners = namedtuple(
        "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
    )

    return GridCorners(
        ur_grid=ur_grid, ul_grid=ul_grid, lr_grid=lr_grid, ll_grid=ll_grid
    )


def get_grid_time_series_at_intersections(grid_data, intersect_weights):
    """Get time series from grid data using sparse intersection weights.

    Time series of grid data are derived via intersection weights of CMLs.
    Please note that it is crucial to have the correct order of dimensions, see
    parameter list below.

    Input can be ndarrays or xarray.DataArrays. If at least one input is a
    DataArray, a DataArray is returned.


    Parameters
    ----------
    grid_data: ndarray or xarray.DataArray
        3-D data of the gridded data we want to extract time series from at the
        given pixel intersection. The order of dimensions must be ('time', 'y', 'x').
        The size in the `x` and `y` dimension must be the same as in the intersection
        weights.
    intersect_weights: ndarray or xarray.DataArray
        3-D data of intersection weights. The order of dimensions must be
        ('cml_id', 'y', 'x'). The size in the `x` and `y` dimension must be the
        same as in the grid data. Intersection weights do not have to be a
        `sparse.array` but will be converted to one internally before doing a
        `sparse.tensordot` contraction.

    Returns
    -------
    grid_intersect_timeseries: ndarray or xarray.DataArray
        The time series for each grid intersection. If at least one of the inputs is
        a xarray.DataArray, a xarray.DataArray is returned. Coordinates are
        derived from the input.
    DataArrays.

    """
    return_a_dataarray = False
    try:
        intersect_weights_coords = intersect_weights.coords
        # from here on we only want to deal with the actual array
        intersect_weights = intersect_weights.data
        return_a_dataarray = True
    except AttributeError:
        pass
    try:
        grid_data_coords = grid_data.coords
        # from here on we only want to deal with the actual array
        grid_data = grid_data.data
        return_a_dataarray = True
    except AttributeError:
        pass

    # Assure that we use a sparse matrix for the weights, because, besides
    # being much faster for large tensordot computation, it can deal with
    # NaN better. If the weights are passed to `sparse.tensordot` as numpy
    # arrays, the value for each time series for a certain point in time is NaN
    # if there is at least one nan in the grid at that point in time. We only
    # want NaN in the time series if the intersection intersects with a NaN grid pixel.
    intersect_weights = sparse.asCOO(intersect_weights, check=False)

    grid_intersect_timeseries = sparse.tensordot(
        grid_data,
        intersect_weights,
        axes=[[1, 2], [1, 2]],
    )

    if return_a_dataarray:
        try:
            dim_0_name = grid_data_coords.dims[0]
            dim_0_values = grid_data_coords[dim_0_name].to_numpy()
        except NameError:
            dim_0_name = "time"
            dim_0_values = np.arange(grid_intersect_timeseries.shape[0])
        try:
            dim_1_name = intersect_weights_coords.dims[0]
            dim_1_values = intersect_weights_coords[dim_1_name].to_numpy()
        except NameError:
            dim_1_name = "cml_id"
            dim_1_values = np.arange(grid_intersect_timeseries.shape[1])
        grid_intersect_timeseries = xr.DataArray(
            data=grid_intersect_timeseries,
            dims=(dim_0_name, dim_1_name),
            coords={dim_0_name: dim_0_values, dim_1_name: dim_1_values},
        )

    return grid_intersect_timeseries
