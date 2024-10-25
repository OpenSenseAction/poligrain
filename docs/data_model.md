# Data model

We enforce the usage of `xarray.Dataset` and `xarray.DataArray` with specific
naming conventions defined in the
[OPENSENSE OS data format conventions](https://github.com/OpenSenseAction/OS_data_format_conventions).
This mainly concerns the naming of the variables that hold the coordinates. By
enforcing this naming convention we simplify all functionality regarding spatial
calculation and plotting.

Note that only longitude and latitude coordinates (in decimal degrees) are
required to meet the
[OPENSENSE OS data format conventions](https://github.com/OpenSenseAction/OS_data_format_conventions).
Additionally we require `x` and `y` (and `site_0_x`, etc. for line data), which
should be projected coordinates, to be present in a `xarray.Dataset`. They are
used for distance calculations. We provide a simple function to do the
projection, but the user is free to choose which projection to use. Since we use
the projected coordinates for distance calculation, it should preserve distances
as good as possible in the region of interest.

## Point data

This is an example of a dataset for point data with the required variables and
their required names.

```
<xarray.Dataset>
Dimensions:    (time: 219168, id: 134)
Coordinates:
  * time       (time) datetime64[ns] 2016-05-01T00:05:00 ...
  * id         (id) <U6 'ams1' 'ams2' 'ams3' ...              # Station ID
    lat        (id) float64 52.31 52.3 52.31 ...              # Latitude in decimal degrees
    lon        (id) float64 4.671 4.675 4.677 4.678 ...       # Longitude in decimal degrees
    x          (id) float64 2.049e+05 2.052e+05 ...           # Projected x coordinates
    y          (id) float64 5.804e+06 5.803e+06 ...           # Projected y coordinates
```

## Line data

When working with CML data we assume the following data structure:

```
<xarray.Dataset> Size: 128kB
Dimensions:       (cml_id: 359, time: 31)
Coordinates: (12/15)
    sublink_id    <U9 36B ...
  * cml_id        (cml_id) int64 3kB 10001 10002 10003 ... 10362 10363 10364
    site_0_lat    (cml_id) float64 3kB 57.7 57.73 57.69 ... 57.65 57.66 57.71
    site_0_lon    (cml_id) float64 3kB 12.0 11.98 11.97 ... 12.12 12.03 12.01
    site_1_lat    (cml_id) float64 3kB 57.7 57.72 57.69 ... 57.66 57.63 57.71
    site_1_lon    (cml_id) float64 3kB 11.99 11.97 11.98 ... 12.14 11.97 11.98
  * time          (time) datetime64[ns] 248B 2015-07-25T12:30:00 ... 2015-07-...
    site_0_x      (cml_id) float64 3kB 6.785e+05 6.776e+05 ... 6.792e+05
    site_0_y      (cml_id) float64 3kB 6.4e+06 6.402e+06 ... 6.394e+06 6.4e+06
    site_1_x      (cml_id) float64 3kB 6.783e+05 6.77e+05 ... 6.778e+05
    site_1_y      (cml_id) float64 3kB 6.399e+06 6.402e+06 ... 6.401e+06
Data variables:
    R             (time, cml_id) float64 89kB ...
```

Here, `site_0_x` and `site_0_y` are projected coordinates of `site_0_lon` and
`site_0_lat`. Typically only the lon-lat coordinates are given. But we rely on
the projected coordinates for distance calculations.

For SML data only the ground site has a lon-lat coordinate pair. With the info
on the longitude of the geostationary satellite it is point to, we can calculate
elevation and azimuth. With a given melting layer height, e.g. taken from
atmospheric model output, we can then derive the path that is passing through
rain, where most of the path attenuation is caused. For a given melting layer
height we can then assign a virtual `site_1_x` and `site_0_y` which use for
plotting, see [PR71](https://github.com/OpenSenseAction/poligrain/pull/71).

## Gridded data

For gridded data, mostly weather radar data in our applications, we assume the
following data structure:

```
<xarray.Dataset> Size: 484kB
Dimensions:          (time: 31, x: 37, y: 48)
Coordinates:
  * time             (time) datetime64[ns] 248B 2015-07-25T12:30:00 ... 2015-...
  * x                (x) float64 296B -1.542e+05 -1.522e+05 ... -8.22e+04
  * y                (y) float64 384B -3.413e+06 -3.415e+06 ... -3.507e+06
    lat              (y, x) float32 7kB 57.21 57.21 57.21 ... 58.06 58.06 58.06
    lon              (y, x) float32 7kB 11.41 11.45 11.48 ... 12.59 12.62 12.66
    xs               (y, x) float64 14kB 6.457e+05 6.478e+05 ... 7.157e+05
    ys               (y, x) float64 14kB 6.343e+06 6.343e+06 ... 6.441e+06
Data variables:
    rainfall_amount  (time, y, x) float64 440kB 0.01078 0.0 ... 0.121 0.05403
```

Here, `xs` and `ys` are the projected coordinates with the same shape as `lon`
and `lat`. Most often only `lon` and `lat` are provided. Note that `x` and `y`
are only 1D arrays. They might define an equidistant 2D xy-grid but that is not
a requirement in our data model. We rely on `xs` and `ys` for distance
calculations.

Note that there are different ways to define the location of grid cells, e.g.
the coordinates can define the grid centroid or the lower left corner. We take
that into account in the grid intersection code in `GridAtLines`. But this is
not yet taken into account in `GridAtPoints`.
