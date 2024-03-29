# Release

The module is used to generate release files for ladim. 


## Usage

Release files are generated by running the `makrel` shell script,

```
makrel <conf> <out>
```

the `ladim_plugins.release` main module,

```
python -m ladim_plugins.release <conf> <out>
```

or by calling the `make_release` function directly,

```
from ladim_plugins.release import make_release
release = make_release(conf, out)
```

The argument `out` is the name of the output file. Defaults to
`None` (no file output, only in-memory return value).

The argument `conf` is a dict or a `.yaml` file. See the comments in
[release.yaml](./release.yaml) for details. 

The function returns a dict with the generated release data. If a file name is given, the data
is written to a tab-delimited text file with no headers. The columns will be in
the same order as the dict returned by the function.

## History

Let output be sorted by release date, as ladim requires date sorting to work
properly (Pål Næverlid Sævik, Nov 2021)

Allow attributes generated from statistical distributions (Pål Næverlid Sævik,
Apr 2021)

Add attributes from geojson files (Pål Næverlid Sævik, May 2022)
 
Created by Pål Næverlid Sævik (2020).
