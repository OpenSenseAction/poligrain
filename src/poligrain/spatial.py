"""Functions for calculating spatial distance, intersections and finding neighbors"""

import pyproj


def project_coordinates(x, y, target_projection, source_projection="EPSG:4326"):
    x, y = pyproj.transform(
        pyproj.Proj(target_projection),
        pyproj.Proj(source_projection),
        x,
        y,
        always_xy=True,
    )


def calc_point_to_point_distances(
    ds_points_a, ds_points_b, project_coordinates_to=None
):
    """Calcualte the distance between two datasets of points"""

    if project_coordinates_to is None:
        try:
            x_a, y_a = ds_points_a.x, ds_points_a.y
            x_b, y_b = ds_points_b.x, ds_points_b.y
        except:
            raise AttributeError(
                "if `x` and `y` are not present in `ds_points` a projection has to be provided as EPSG string via `project_coordinates_to"
            )
    else:
        x_a, y_a = project_coordinates_to(
            ds_points_a.x,
            ds_points_a.y,
            target_projection=project_coordinates_to,
        )
        x_b, y_b = project_coordinates_to(
            ds_points_b.x,
            ds_points_b.y,
            target_projection=project_coordinates_to,
        )

    # calc distance...
