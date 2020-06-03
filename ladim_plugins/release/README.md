# Release

The module is used to generate release files for ladim. 


## Usage

Release files are generated using the `make_release` function:

```
from ladim_plugins.release import make_release
release = make_release(conf)
```

The argument `conf` is a dict with the following keywords: 

-   *num*: Number of particles
-   *date*: Date and time for the release, ISO format (YYYY-MM-DD hh:mm:ss).
    Either a single date or a list of the form [start, stop], which represents
    a date span.
-   *location*: Release position. Several forms of this argument is possible:
    1.  A list of the form [lon, lat], representing a point
    2.  A list of lists, of the form [[lon_1, lon_2, ...], [lat_1, lat_2, ...]],
        representing a polygon
    3.  A two-element list of lists of lists, of the form [[[lo1A, lo2A, ...],
        [lo1B, lo2B, ...], ...], [[lo1A, lo2A, ...], [lo1B, lo2B, ...], ...]]
        which represents a collection of polygons.
    4.  A dict of the form `{'center': [lon, lat], 'offset': [[lon_1, lon_2, 
        ...], [lat_1, lat_2, ...]]}`,
        where `offset` is a polygon, specified as a metric offset from the
        `center` location. The `offset` variable could also be a collection
        of polygons, using a two-element list of lists of lists.
-   *depth*: Release depth in meters, defaults to zero. Two forms of the
    argument is possible:
    1.  A single number, representing a single depth
    2.  A list of the form [min, max], representing an even distribution of
        particles over the given depth range.
-   *attrs*: A dictionary (key-value pairs) of particle attributes. Attribute
    values may be single values (= all particles are equal), or one value per
    particle, or a function / function name with the number of particles as its
    sole argument.
-   *file*: Name of the output file (or handle). Defaults to `None` (no file
    output).

Alternatively, `conf` can be a list of dicts, representing the union of several
independent release configurations. Finally, `conf` can also be the name of a
`.yaml` file containing the configuration keywords.

Returns a dict with the following fields (in order): `release_time`, `lon`,
`lat`, `Z`, followed by the attribute fields. If a file name is given, the data
is written to a tab-delimited text file with to headers. The columns will be in
the same order as the dict returned by the function.

## History

Created by Pål Næverlid Sævik (2020).
