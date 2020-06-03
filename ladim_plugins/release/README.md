# Release

The module is used to generate release files for ladim. 


## Usage

Release files are generated using the `make_release` function:

```
from ladim_plugins.release import make_release
release = make_release(conf)
```

The argument `conf` is a dict with the following keywords: 

-   *start_date*: Date and time for first release, ISO format (YYYY-MM-DD hh:mm:ss)
-   *stop_date*: Date and time for final release. Defaults to `start_date`
-   *num*: Number of particles
-   *location*: Release position. Two forms of this argument is possible:
    1.  A list of the form [lon, lat], representing a point
    2.  A list of lists, of the form [[lon_1, lon_2, ...], [lat_1, lat_2, ...]],
        representing a polygon
-   *depth*: Release depth in meters, defaults to zero. Two forms of the
    argument is possible:
    1.  A single number, representing a single depth
    2.  A list of the form [min, max], representing an even distribution of
        particles over the given depth range.
-   *attrs*: A dictionary (key-value pairs) of particle attributes. Attribute
    values may be single values (= all particles are equal), or one value per
    particle, or a function / function name with no arguments.
-   *file*: Name of the output file (or handle). Defaults to `None` (no file output).

Alternatively, `conf` can be a list of dicts, representing the union of several
independent release configurations. Finally, `conf` can also be the name of a
`.yaml` file containing the configuration keywords.


## History

Created by Pål Næverlid Sævik (2020).
