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

...to be added

## Gridded data

...to be added
