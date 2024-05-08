import unittest
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import poligrain as plg


class TestSparseIntersectWeights(unittest.TestCase):
    def test_creation_of_xarray_dataarray(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(12))

        x1_list = [0, 0]
        y1_list = [0, 0]
        x2_list = [0, 9]
        y2_list = [9, 9]
        cml_id_list = ["abc1", "cde2"]

        da_intersect_weights = (
            plg.spatial.calc_sparse_intersect_weights_for_several_cmls
        )(
            x1_line=x1_list,
            y1_line=y1_list,
            x2_line=x2_list,
            y2_line=y2_list,
            cml_id=cml_id_list,
            x_grid=x_grid,
            y_grid=y_grid,
        )

        for x1, y1, x2, y2, cml_id in zip(
            x1_list, y1_list, x2_list, y2_list, cml_id_list
        ):
            expected = plg.spatial.calc_intersect_weights(
                x1_line=x1,
                y1_line=y1,
                x2_line=x2,
                y2_line=y2,
                x_grid=x_grid,
                y_grid=y_grid,
            )
            np.testing.assert_array_almost_equal(
                expected, da_intersect_weights.sel(cml_id=cml_id).to_numpy()
            )


class TestIntersectWeights(unittest.TestCase):
    def test_with_simple_grid(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0, 0
        x2, y2 = 0, 9

        intersec_weights = plg.spatial.calc_intersect_weights(
            x1_line=x1, y1_line=y1, x2_line=x2, y2_line=y2, x_grid=x_grid, y_grid=y_grid
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        x1, y1 = 0, 0
        x2, y2 = 9, 9

        intersec_weights = plg.spatial.calc_intersect_weights(
            x1_line=x1, y1_line=y1, x2_line=x2, y2_line=y2, x_grid=x_grid, y_grid=y_grid
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05555556],
                ]
            ),
        )

    def test_with_simple_grid_location_lower_left(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0.5, 0
        x2, y2 = 0.5, 9

        intersec_weights = plg.spatial.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid,
            grid_point_location="lower_left",
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        x1, y1 = 0.5, 0.5
        x2, y2 = 9.5, 9.5

        intersec_weights = plg.spatial.calc_intersect_weights(
            x1_line=x1,
            y1_line=y1,
            x2_line=x2,
            y2_line=y2,
            x_grid=x_grid,
            y_grid=y_grid,
            grid_point_location="lower_left",
        )

        assert intersec_weights.sum() == 1.0

        np.testing.assert_array_almost_equal(
            intersec_weights,
            np.array(
                [
                    [0.05555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11111111, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05555556],
                ]
            ),
        )

    def test_unknown_grid_location(self):
        x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))

        x1, y1 = 0.5, 0
        x2, y2 = 0.5, 9
        with pytest.raises(
            ValueError, match="`grid_point_location` = upper_middle not implemented"
        ):
            plg.spatial.calc_intersect_weights(
                x1_line=x1,
                y1_line=y1,
                x2_line=x2,
                y2_line=y2,
                x_grid=x_grid,
                y_grid=y_grid,
                grid_point_location="upper_middle",
            )


class TestCalcGridCorners(unittest.TestCase):
    def test_location_at_grid_center(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        result = plg.spatial._calc_grid_corners_for_center_location(grid=grid)

        GridCorners = namedtuple(
            "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
        )
        expected = GridCorners(
            ur_grid=np.stack([x_grid + 0.5, y_grid + 0.5], axis=2),
            ul_grid=np.stack([x_grid - 0.5, y_grid + 0.5], axis=2),
            lr_grid=np.stack([x_grid + 0.5, y_grid - 0.5], axis=2),
            ll_grid=np.stack([x_grid - 0.5, y_grid - 0.5], axis=2),
        )

        np.testing.assert_almost_equal(result.ur_grid, expected.ur_grid)
        np.testing.assert_almost_equal(result.ul_grid, expected.ul_grid)
        np.testing.assert_almost_equal(result.lr_grid, expected.lr_grid)
        np.testing.assert_almost_equal(result.ll_grid, expected.ll_grid)

    def test_location_at_lower_left(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        result = plg.spatial._calc_grid_corners_for_lower_left_location(grid=grid)

        GridCorners = namedtuple(
            "GridCorners", ["ur_grid", "ul_grid", "lr_grid", "ll_grid"]
        )
        expected = GridCorners(
            ur_grid=np.stack([x_grid + 1.0, y_grid + 1.0], axis=2),
            ul_grid=np.stack([x_grid, y_grid + 1.0], axis=2),
            lr_grid=np.stack([x_grid + 1.0, y_grid], axis=2),
            ll_grid=np.stack([x_grid, y_grid], axis=2),
        )

        np.testing.assert_almost_equal(result.ur_grid, expected.ur_grid)
        np.testing.assert_almost_equal(result.ul_grid, expected.ul_grid)
        np.testing.assert_almost_equal(result.lr_grid, expected.lr_grid)
        np.testing.assert_almost_equal(result.ll_grid, expected.ll_grid)

    def test_location_at_lower_left_descending_x_error(self):
        x_grid, y_grid = np.meshgrid(np.arange(20, 10, -1), np.arange(50, 70, 1))
        grid = np.stack([x_grid, y_grid], axis=2)

        with pytest.raises(ValueError, match="x values must be ascending along axis 1"):
            plg.spatial._calc_grid_corners_for_lower_left_location(grid=grid)

    def test_location_at_lower_left_descending_y_error(self):
        x_grid, y_grid = np.meshgrid(np.arange(10, 20, 1), np.arange(70, 50, -1))
        grid = np.stack([x_grid, y_grid], axis=2)

        with pytest.raises(ValueError, match="y values must be ascending along axis 0"):
            plg.spatial._calc_grid_corners_for_lower_left_location(grid=grid)


def get_grid_intersect_ts_test_data():
    grid_data = np.tile(
        np.expand_dims(np.arange(10, dtype="float"), axis=[1, 2]), (1, 4, 4)
    )
    grid_data[0, 0, 1] = np.nan
    # fmt: off
    intersect_weights = np.array(
        [
            [[0.25, 0, 0, 0],
             [0.25, 0, 0, 0],
             [0.25, 0, 0, 0],
             [0.25, 0, 0, 0]],
            [[0, 0.25, 0.25, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
        ]
    )
    # fmt: on
    expected = np.array(
        [
            [0.0, np.nan],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
            [7.0, 3.5],
            [8.0, 4.0],
            [9.0, 4.5],
        ]
    )
    return grid_data, intersect_weights, expected


class TestGetGridTimeseries(unittest.TestCase):
    def test_numpy_grid_numpy_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()

        result = plg.spatial.get_grid_time_series_at_intersections(
            grid_data=grid_data,
            intersect_weights=intersect_weights,
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_dataarray_grid_numpy_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()
        time = pd.date_range("2017-01-01", "2017-01-10")
        da_grid_data = xr.DataArray(
            data=grid_data,
            dims=("time", "y", "x"),
            coords={"time": time},
        )

        result = plg.spatial.get_grid_time_series_at_intersections(
            grid_data=da_grid_data,
            intersect_weights=intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.time.values, time)
        assert result.dims == ("time", "cml_id")

    def test_numpy_grid_dataarray_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()
        cml_ids = ["cml_1", "cml_2"]
        da_intersect_weights = xr.DataArray(
            data=intersect_weights,
            dims=("cml_id", "y", "x"),
            coords={"cml_id": cml_ids},
        )

        result = plg.spatial.get_grid_time_series_at_intersections(
            grid_data=grid_data,
            intersect_weights=da_intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.cml_id.values, cml_ids)
        assert result.dims == ("time", "cml_id")

    def test_dataarray_grid_dataarray_weights(self):
        grid_data, intersect_weights, expected = get_grid_intersect_ts_test_data()

        time = pd.date_range("2017-01-01", "2017-01-10")
        da_grid_data = xr.DataArray(
            data=grid_data,
            dims=("time", "y", "x"),
            coords={"time": time},
        )

        cml_ids = ["cml_1", "cml_2"]
        da_intersect_weights = xr.DataArray(
            data=intersect_weights,
            dims=("cml_id", "y", "x"),
            coords={"cml_id": cml_ids},
        )

        result = plg.spatial.get_grid_time_series_at_intersections(
            grid_data=da_grid_data,
            intersect_weights=da_intersect_weights,
        )
        np.testing.assert_array_almost_equal(result.data, expected)

        np.testing.assert_array_equal(result.cml_id.values, cml_ids)
        np.testing.assert_array_equal(result.time.values, time)

        assert result.dims == ("time", "cml_id")


ds_gauge = xr.Dataset(
    data_vars={
        "rainfall_amount": (("id", "time"), np.reshape(np.arange(1, 13), (3, 4))),
    },
    coords={
        "id": ("id", ["g1", "g2", "g3"]),
        "time": ("time", np.arange(0, 4)),
        "x": ("id", [0, 1, 1]),
        "y": ("id", [0, 0, 1]),
        "lon": ("id", [3.5, 4.1, 5.2]),
        "lat": ("id", [50.1, 50.1, 51.2]),
    },
)


def test_get_point_xy():
    x, y = plg.spatial.get_point_xy(ds_points=ds_gauge)
    assert x.data == pytest.approx(np.array([0, 1, 1]))
    assert y.data == pytest.approx(np.array([0, 0, 1]))

    # check for case with only one point
    x, y = plg.spatial.get_point_xy(ds_points=ds_gauge.isel(id=0))
    assert x.data == pytest.approx(
        np.array(
            [
                0,
            ]
        )
    )
    assert y.data == pytest.approx(
        np.array(
            [
                0,
            ]
        )
    )


def test_project_point_coordinates():
    lon, lat = ds_gauge.lon, ds_gauge.lat

    # With default source_projections
    x, y = plg.spatial.project_point_coordinates(
        x=lon, y=lat, target_projection="EPSG:25832"
    )
    x_expected = np.array([106756.46571167826, 149635.93311767105, 234545.23195888632])
    y_expected = np.array([5564249.372223592, 5561255.584168306, 5678930.9034550935])
    assert x.data == pytest.approx(x_expected, abs=1e-9)
    assert y.data == pytest.approx(y_expected, abs=1e-9)

    # With different source and targe projection, using the opposite direction
    # of the test above from UTM 32N to WGS 80
    x_source = xr.DataArray(
        data=np.array([106756.46571167826, 149635.93311767105, 234545.23195888632]),
        coords={"id": ds_gauge.id.data},
    )
    y_source = xr.DataArray(
        data=np.array([5564249.372223592, 5561255.584168306, 5678930.9034550935]),
        coords={"id": ds_gauge.id.data},
    )

    x, y = plg.spatial.project_point_coordinates(
        x=x_source,
        y=y_source,
        source_projection="EPSG:25832",
        target_projection="EPSG:4326",
    )

    x_expected = lon
    y_expected = lat
    assert x.data == pytest.approx(x_expected, abs=1e-6)
    assert y.data == pytest.approx(y_expected, abs=1e-6)

    # Check that returned DataArray has correct ids
    assert list(x.id.data) == ["g1", "g2", "g3"]
    assert list(y.id.data) == ["g1", "g2", "g3"]


def test_get_closest_points_to_point():
    closest_neighbors = plg.spatial.get_closest_points_to_point(
        ds_points=ds_gauge.sel(id=["g2", "g3"]),
        ds_points_neighbors=ds_gauge,
        max_distance=1.1,
        n_closest=5,
    )
    expected_distances = np.array(
        [[0.0, 1.0, 1.0, np.inf, np.inf], [0.0, 1.0, np.inf, np.inf, np.inf]]
    )
    nan = np.float64(np.nan)
    expected_neighbor_ids = np.array(
        [["g2", "g3", "g1", nan, nan], ["g3", "g2", nan, nan, nan]], dtype=object
    )

    assert closest_neighbors.distance.data == pytest.approx(
        expected_distances, abs=1e-6
    )
    assert (
        closest_neighbors.neighbor_id.data[expected_distances != np.inf]
        == expected_neighbor_ids[expected_distances != np.inf]
    ).all()
    assert np.isnan(
        closest_neighbors.neighbor_id.data[expected_distances == np.inf].astype(float)
    ).all()

    # check with different parameters
    closest_neighbors = plg.spatial.get_closest_points_to_point(
        ds_points=ds_gauge.sel(id=["g2", "g3"]),
        ds_points_neighbors=ds_gauge,
        max_distance=2,
        n_closest=4,
    )
    assert closest_neighbors.distance.data[1, 2] == pytest.approx(1.414213562, abs=1e-6)
    assert closest_neighbors.distance.data.shape == (2, 4)

    # check case with n_closest=1
    closest_neighbors = plg.spatial.get_closest_points_to_point(
        ds_points=ds_gauge.sel(id=["g2", "g3"]),
        ds_points_neighbors=ds_gauge,
        max_distance=2,
        n_closest=1,
    )
    assert closest_neighbors.distance.data.shape == (2, 1)


def test_calc_point_to_point_distances():
    distance_matrix = plg.spatial.calc_point_to_point_distances(
        ds_points_a=ds_gauge, ds_points_b=ds_gauge.sel(id=["g2", "g3"])
    )
    expected = np.array(
        [
            [1, np.sqrt(2)],
            [0, 1],
            [1, 0],
        ]
    )
    assert distance_matrix.data == pytest.approx(expected, abs=1e-6)
    assert list(distance_matrix.id.data) == ["g1", "g2", "g3"]
    assert list(distance_matrix.id_neighbor.data) == ["g2", "g3"]
