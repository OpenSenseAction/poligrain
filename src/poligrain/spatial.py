"""Functions for calculating spatial distance, intersections and finding neighbors."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pyproj
import scipy
import sparse
import xarray as xr
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon


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
    assert ds_points.x.dims == ds_points.y.dims
    if len(ds_points.x.dims) > 1:
        msg = f"x and y should be 1D or 0D, but are {len(ds_points.x.dims)}D."
        raise ValueError(msg)
    if len(ds_points.x.dims) == 0:
        return (
            ds_points.x.expand_dims(dim={"id": 1}),
            ds_points.y.expand_dims(dim={"id": 1}),
        )
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


def get_closest_points_to_point(
    ds_points: xr.DataArray | xr.Dataset,
    ds_points_neighbors: xr.DataArray | xr.Dataset,
    max_distance: float,
    n_closest: int,
) -> xr.Dataset:
    """Get the closest points for given point locations.

    Note that both datasets that are passed as input have to have the
    variables `x` and `y` which should be projected coordinates that preserve
    lengths as good as possible.

    Parameters
    ----------
    ds_points : xr.DataArray | xr.Dataset
        This is the dataset for which the nearest neighbors will be looked up. That is,
        for each point location in this dataset the nearest neighbors from
        `ds_points_neighbors` will be returned.
    ds_points_neighbors : xr.DataArray | xr.Dataset
        This is the dataset from which the nearest neighbors will be looked up.
    max_distance : float
        The allowed distance of neighbors has to be smaller than `max_distance`.
        The unites are the units used for the projected coordinates `x` and
        `y` in the two datasets.
    n_closest : int
        The maximum number of nearest neighbors to be returned.

    Returns
    -------
    xr.Dataset
        A dataset which has `distance` and `neighbor_id` as variables along the
        dimensions `id`, taken from `ds_points` and `n_closest`. The unit of the
        distance follows from the unit of the projected coordinates of the input
        datasets. The `neighbor_id` entries for point locations that are further
        away then `max_distance` are set to None. The according distances are np.inf.
    """
    x, y = get_point_xy(ds_points)
    x_neighbors, y_neighbors = get_point_xy(ds_points_neighbors)
    tree_neighbors = scipy.spatial.KDTree(
        data=list(zip(x_neighbors.values, y_neighbors.values))
    )
    distances, ixs = tree_neighbors.query(
        list(zip(x.values, y.values)),
        k=n_closest,
        distance_upper_bound=max_distance,
    )
    # Note that we need to transpose to have the extended dimension, which
    # is the one for `n_closest` in the xr.Dataset later on, as last dimension.
    # To preserve an existing 2D array we need to transpose also before applying
    # `at_least2d` so that we have two times a transpose and get the origianl 2D
    # array.
    distances = np.atleast_2d(distances.T).T
    ixs = np.atleast_2d(ixs.T).T

    # Make sure that we have 'id' dimension in case only one station is there
    # in ds_points_neighbors because e.g. ds.isel(id=0) was used to subset.
    if "id" not in ds_points_neighbors.dims:
        ds_points_neighbors = ds_points_neighbors.expand_dims("id")

    # Where neighboring station are further away than max_distance the ixs are
    # filled with the value n, the length of the neighbor dataset. We want to
    # return None as ID in the cases the index is n. For this we must pad the
    # array of neighbor IDs with None. Because padding is always done symmetrically,
    # we have to slice off the padding on the left. But we cannot pad directly with
    # None. We get NaN even if we do `.pad(constant_values=None)`. Hence, we fill
    # the NaNs we get from padding with None afterwards.
    # This way the index n points to this last entry on the right which we want to
    # be None.
    id_neighbors_nan_padded = ds_points_neighbors.id.pad(
        id=1,
        mode="constant",
    ).isel(id=slice(1, None))
    id_neighbors_nan_padded = id_neighbors_nan_padded.where(
        ~id_neighbors_nan_padded.isnull(),  # noqa: PD003
        None,
    )
    neighbor_ids = id_neighbors_nan_padded.data[ixs]

    # Make sure that `id` dimension is not 0, which happens if input only
    # has one station e.g. because ds.isel(id=0) was used to subset.
    if ds_points.id.ndim == 0:
        ds_points = ds_points.expand_dims("id")

    return xr.Dataset(
        data_vars={
            "distance": (("id", "n_closest"), distances),
            "neighbor_id": (("id", "n_closest"), neighbor_ids),
        },
        coords={"id": ds_points.id.data},
    )


def calc_point_to_point_distances(
    ds_points_a: xr.DataArray | xr.Dataset, ds_points_b: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Calculate the distance between the point coordinates of two datasets.

    Note that both datasets that are passed as input have to have the
    variables `x` and `y` which should be projected coordinates that preserve
    lengths as good as possible.

    Parameters
    ----------
    ds_points_a : xr.DataArray | xr.Dataset
        One dataset of points.
    ds_points_b : xr.DataArray | xr.Dataset
        The other dataset of points.

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


class GridAtLines:
    """Get path-averaged grid values along lines.

    For each line, e.g. a CML path, in `ds_line_data` the grid intersections are
    calculated and stored as sparse matrix during initialization. Via `__call__`
    the time series of path-averaged grid values for each line can be calculated.

    Note that `da_gridded_data` and `ds_line_data` have to contain the correct
    coordinate variable names, see below.

    Parameters
    ----------
    da_gridded_data
        The gridded data, typically rainfall fields from weather radar. It has to
        contain `lon` and `lat` variables with coordinates as 2D matrix.
    ds_line_data
        The line data, typically from CMLs. It has to contain lon and lat
        coordinates for site_0 and site_1 according to the OPENSENSE data
        format conventions.
    grid_point_location
        The location of the grid point for which the coordinates are given. Can
        be "center" or "lower_right". Default is "center".

    """

    def __init__(
        self,
        da_gridded_data: xr.DataArray | xr.Dataset,
        ds_line_data: xr.DataArray | xr.Dataset,
        grid_point_location: str = "center",
    ):
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

    def __call__(self, da_gridded_data: xr.DataArray) -> xr.DataArray:
        """Calculate path-averaged grid values along lines.

        Parameters
        ----------
        da_gridded_data
            3-D data of the gridded data. The order of dimensions must be
            ('time', 'y', 'x'). The grid must be the same as in the initialization
            of the current object because the intersection weights are calculated
            at initialization.

        Returns
        -------
        gridded_data_along_line
            The time series for each grid intersection of each line. The IDs for each
            line are taken from `ds_line_data` in `__init__`.
        """
        gridded_data_along_line = get_grid_time_series_at_intersections(
            grid_data=da_gridded_data,
            intersect_weights=self.intersect_weights,
        )

        gridded_data_along_line["time"] = da_gridded_data.time
        gridded_data_along_line["site_0_lon"] = self.intersect_weights.site_0_lon
        gridded_data_along_line["site_1_lat"] = self.intersect_weights.site_1_lat
        gridded_data_along_line["site_1_lon"] = self.intersect_weights.site_1_lon
        gridded_data_along_line["site_0_lat"] = self.intersect_weights.site_0_lat

        return gridded_data_along_line


class GridAtPoints:
    """Get grid values at points or in their neighborhood.

    This class is based on `wradblib.adjust.RawAtObs` which already implements
    all required functionality. To make usage simpler and equivalent to `GridAtLines`,
    we just provide a wrapper around the `RawAtObs` code that was copy-pasted to not
    rely on a import of `wradlib`. In addition we use `xarray.DataArray` or
    `xarray.Dataset` as input to avoid reshaping and passing point and grid coordinates.

    Note that `da_gridded_data` and `da_point_data` have to contain the correct
    coordinate variable names, see below.

    Parameters
    ----------
    da_gridded_data
        The gridded data, typically rainfall fields from weather radar. It has to
        contain `lon` and `lat` variables with coordinates as 2D matrix.
    da_point_data
        The point data, typically from rain gauges. The coordinates must be given
        as `lon` and `lat` variable according to the OPENSENSE data format conventions.
    nnear
        Number of neighbors which should be considered in the vicinity of each
        point in obs. Note that this is using a nearest-neighbor-lookup and does
        not guarantee that e.g. `nnear=9` results in a 3x3 grid with the central
        pixel in the middle.
    stat
        Name of stat function to be used to derive one value from all neighboring
        grid cells in case that `nnear > 1`. Default is "best".
    """

    def __init__(
        self,
        da_gridded_data: xr.DataArray | xr.Dataset,
        da_point_data: xr.DataArray | xr.Dataset,
        nnear: int,
        stat: str = "best",
    ):
        # Get radar pixel coordinates as (N, 2) array
        x_grid = da_gridded_data.lon.to_numpy()
        y_grid = da_gridded_data.lat.to_numpy()
        assert x_grid.shape == y_grid.shape
        xy_grid = np.array(list(zip(x_grid.flatten(), y_grid.flatten())))

        # Initialize function to get grid values at points
        xy_points = np.stack([da_point_data.lon, da_point_data.lat], axis=1)

        # copy-paste code from wradlib.adjust.RawAtObs.__init__
        obs_coords = xy_points
        raw_coords = xy_grid
        self.statfunc = _get_statfunc(stat)
        self.raw_ix = _get_neighbours_ix(obs_coords, raw_coords, nnear)

    def __call__(
        self, da_gridded_data: xr.DataArray, da_point_data: xr.DataArray
    ) -> xr.DataArray:
        """Return grid values at points.

        Note that here, both inputs must be a `xr.DataArray` with a `time` dimension.
        We require `da_point_data` as input because it is needed for the calculation
        in case of `nnear > 1` and `stat="best"`.

        Parameters
        ----------
        da_gridded_data
            The gridded data on the same grid as used in `__init__`. It has to have a
            time dimension over which it is iterated.
        da_point_data : xr.DataArray
            The point data with the same coordinates and exact same ordering as used
            in `__init__`. There has to be a `time` dimension. The time stamps from
            `da_gridded_data` are selected. Hence, `da_point_data` and `da_gridded_data`
            should have matching time stamps. The time period can be longer or shorter,
            though. Non matching time stamps are just not considered.

        Returns
        -------
            The time series of grid values at each point. The IDs for each
            point are taken from `da_point_data`.

        """
        gridded_data_at_point_list = []
        for t in da_gridded_data.time.to_numpy():
            da_gridded_data_t = da_gridded_data.sel(time=t)
            da_point_data_t = da_point_data.sel(time=t)

            # copy-paste code from wradlib.adjust.RawAtObs.__call__
            raw = da_gridded_data_t.to_numpy().flatten()
            obs = da_point_data_t.to_numpy()
            # get the values of the raw neighbours of obs
            raw_neighbs = raw[self.raw_ix]
            # and summarize the values of these neighbours
            # by using a statistics option
            # (only needed in case nnear > 1, i.e. multiple neighbours
            # per observation location)
            if raw_neighbs.ndim > 1:
                gridded_data_at_point = self.statfunc(obs, raw_neighbs)
            else:
                gridded_data_at_point = raw_neighbs

            gridded_data_at_point_list.append(gridded_data_at_point)

        gridded_data_at_points = np.array(gridded_data_at_point_list)
        if da_point_data.dims[0] != "time":
            gridded_data_at_points = np.transpose(gridded_data_at_points)

        return xr.DataArray(
            data=gridded_data_at_points,
            dims=da_point_data.dims,
            coords={
                "time": da_gridded_data.time,
                "lon": da_point_data.lon,
                "lat": da_point_data.lat,
                "id": da_point_data.id,
            },
        )


def _get_neighbours_ix(obs_coords, raw_coords, nnear):
    """Return ``nnear`` neighbour indices per ``obs_coords`` coordinate pair.

    Parameters
    ----------
    obs_coords : :py:class:`numpy:numpy.ndarray`
        array of float of shape (num_points,ndim)
        in the neighbourhood of these coordinate pairs we look for neighbours
    raw_coords : :py:class:`numpy:numpy.ndarray`
        array of float of shape (num_points,ndim)
        from these coordinate pairs the neighbours are selected
    nnear : int
        number of neighbours to be selected per coordinate ``obs_coords``

    """
    # plant a tree
    tree = scipy.spatial.cKDTree(raw_coords)
    # return nearest neighbour indices
    return tree.query(obs_coords, k=nnear)[1]


def _get_statfunc(funcname):
    """Return a function that corresponds to parameter ``funcname``.

    This is a copy-paste function from `wradlib.adjust` that we need
    for our implementation that is similar to their `RawAtObs`.

    Parameters
    ----------
    funcname : str
        a name of a numpy function OR another option known by _get_statfunc
        Potential options: 'mean', 'median', 'best'

    """
    if funcname == "best":
        newfunc = best
    else:
        try:
            # try to find a numpy function which corresponds to <funcname>
            func = getattr(np, funcname)

            # To be compatible with `best(x, y)` we wrap `func` and
            # do not use `x`.
            def newfunc(x, y):  # pylint: disable=unused-argument, # noqa: ARG001
                return func(y, axis=1)

        except Exception as err:
            raise NameError(f"Unknown function name option: {funcname!r}") from err  # noqa: EM102
    return newfunc


def best(x, y):
    """Find the values of y which corresponds best to x.

    If x is an array, the comparison is carried out for each element of x.

    This is a copy-paste function from `wradlib.adjust` that we need
    for our implementation that is similar to their `RawAtObs`.

    Parameters
    ----------
    x : float | :py:class:`numpy:numpy.ndarray`
        float or 1-d array of float
    y : :py:class:`numpy:numpy.ndarray`
        array of float

    Returns
    -------
    output : :py:class:`numpy:numpy.ndarray`
        1-d array of float with length len(y)

    """
    if type(x) == np.ndarray:  # pylint: disable=unidiomatic-typecheck
        if x.ndim != 1:
            raise ValueError("`x` must be a 1-d array of floats or a float.")  # noqa: EM101
        if len(x) != len(y):
            raise ValueError(
                f"Length of `x` ({len(x)}) and `y` ({len(y)}) must be equal."  # noqa: EM102
            )
    if type(y) == np.ndarray:  # pylint: disable=unidiomatic-typecheck
        if y.ndim > 2:
            raise ValueError("'y' must be 1-d or 2-d array of floats.")  # noqa: EM101
    else:
        raise ValueError("`y` must be 1-d or 2-d array of floats.")  # noqa: EM101
    x = np.array(x).reshape((-1, 1))
    if y.ndim == 1:
        y = np.array(y).reshape((1, -1))
        axis = None
    else:
        axis = 1
    return y[np.arange(len(y)), np.argmin(np.abs(x - y), axis=axis)]


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
        The location of the grid point for which the coordinates are given. Can
        be "center" or "lower_right". Default is "center".

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
        The location of the grid point for which the coordinates are given. Can
        be "center" or "lower_right". Default is "center".
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
        msg = f"`grid_point_location` = '{grid_point_location}' not implemented"
        raise ValueError(msg)

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


def get_closest_points_to_line(ds_cmls, ds_gauges, max_distance, n_closest):
    """Get closest points to line.

    Finds n closest points from a CML within given max distance. Note that the
    function guarantees that all returned points are within max distance to
    the CML, not that all points that are within max distance are returned. Uses
    KDTree for fast processing of large datasets.

    Parameters
    ----------
    ds_cmls: xarray.Dataset
        Dataset of line data using the OpenSense naming convention for CMLs. It
        must contain the coordinate cml_id with the cml names. It must also
        contain projected coordinates site_0_y, site_0_x, site_1_y and site_1_x
        as well as the CML length.
    ds_gauges: xarray.Dataset
        Dataset of point data using the OpenSense data format conventions for PWS.
        The dataset must contain the coordinate 'id' with the PWS names. It must
        also contain projected coordinates x and y.
    max_distance: float
        Maximum distance a point can have to the CML, measured as the smallest
        distance from the point to the line. Points outside this range is not
        considered close to the CML.
    n_closest: int
        Maximum number of points that are returned.

    Returns
    -------
    closest_gauges: xarray.Dataset
        Dataset with CML ids and corresponding n_closest point names and distance.
        If a CML has less that "n_closest" nearby points, the remaining entries
        in variable "distance" are filled with np.inf and the remaining entries
        in variable "id_neighbor" are filled with None.

    """
    # Add dim "cml_id" if not present, for instance if user selects only 1 cml
    if "cml_id" not in ds_cmls.dims:
        ds_cmls = ds_cmls.expand_dims("cml_id")

    # Add dim "id" if not present, for instance if user selects only 1 gauge
    if "id" not in ds_gauges.dims:
        ds_gauges = ds_gauges.expand_dims("id")

    # Transfer raingauge and CML coordinates to numpy, for faster access in loop
    coords_cml_a = np.hstack(
        [ds_cmls.site_0_y.data.reshape(-1, 1), ds_cmls.site_0_x.data.reshape(-1, 1)]
    )
    coords_cml_b = np.hstack(
        [ds_cmls.site_1_y.data.reshape(-1, 1), ds_cmls.site_1_x.data.reshape(-1, 1)]
    )
    coords_gauge = np.hstack(
        [ds_gauges.y.data.reshape(-1, 1), ds_gauges.x.data.reshape(-1, 1)]
    )

    # Store half length of CML for fast lookup when setting max_distance
    cml_half_lengths = np.atleast_1d(ds_cmls.length.data / 2)

    # Calculate CML midpoints by using the average of site a and b
    coords_cml = np.hstack(
        [
            ((coords_cml_a[:, 0] + coords_cml_b[:, 0]) / 2).reshape(-1, 1),
            ((coords_cml_a[:, 1] + coords_cml_b[:, 1]) / 2).reshape(-1, 1),
        ]
    )

    # Array for storing name of gauges close to CML
    list_gauges = np.full([ds_cmls.cml_id.size, n_closest], None)

    # Array for storing distances between nearby gauges and CML
    cml_gauge_dist = np.full([ds_cmls.cml_id.size, n_closest], np.inf)

    # Create KDTree object for all gauges
    kd_tree = KDTree(coords_gauge)

    for i in range(len(ds_cmls.cml_id)):
        # Query KDTree for all points within max_distance +
        # cml_half_lengths from the CML midpoint
        ind_nearest_gauges = kd_tree.query_ball_point(
            coords_cml[i],
            cml_half_lengths[i] + max_distance,
        )

        # Ensure this is always an array
        ind_nearest_gauges = np.atleast_1d(ind_nearest_gauges)

        # Create line object for current CML
        line = LineString([coords_cml_a[i], coords_cml_b[i]])

        # Calculate the precise distances to nearby gauges
        distances = np.array(
            [
                line.distance(Point(coords_gauge[ind]))
                if ind != len(coords_gauge)
                else np.inf  # set to np.inf if outside max_distance
                for ind in ind_nearest_gauges
            ]
        )

        # Get the sorted indices
        dist_sort_ind = np.argsort(distances)

        # Select the n_closest gauges if more than 'n_closest' near cml
        if dist_sort_ind.size > n_closest:
            dist_sort_ind = dist_sort_ind[:n_closest]

        # Get the indices of the corresponding gauges
        gauge_ind = ind_nearest_gauges[dist_sort_ind]

        # store results if there are any nearby gauges
        if gauge_ind.size > 0:
            list_gauges[i, : dist_sort_ind.size] = ds_gauges.id.data[gauge_ind]
            cml_gauge_dist[i, : dist_sort_ind.size] = distances[dist_sort_ind]

    # Set IDs above max_distance to None
    list_gauges[cml_gauge_dist > max_distance] = None

    # Set distances above max_distance to inf
    cml_gauge_dist[cml_gauge_dist > max_distance] = np.inf

    # Create xarray object showing name and distance from cml to nearest points
    return xr.Dataset(
        data_vars={
            "distance": (("cml_id", "n_closest"), cml_gauge_dist),
            "neighbor_id": (("cml_id", "n_closest"), list_gauges),
        },
        coords={
            "cml_id": ds_cmls.cml_id.data,
        },
    )
