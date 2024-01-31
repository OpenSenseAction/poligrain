import numpy as np
import pytest
import xarray as xr

import poligrain as plg

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

    # TODO: Add variants with other args


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
