import xarray as xr
import numpy as np
from ladim_plugins.release.makrel import degree_diff_to_metric
import logging
import glob


logger = logging.getLogger(__name__)


def ladim_raster(particle_dset, grid_dset, weights=(None,)):
    """
    Convert LADiM output data to raster format.

    :param particle_dset: Particle file from LADiM (file name or xarray dataset)

    :param grid_dset: NetCDF file containing bin centers (file name or xarray dataset).

        * Any coordinate variable in the grid file which match the name of a variable
          in the LADiM file is used as bin centers, in the order which they appear in the
          grid file.

        * Bin edges are assumed to be halfway between bin centers, if not specified
          otherwise by a `bounds` attribute.

        * If the dataset includes a CF-compliant grid mapping, the output dataset will
          include a variable `cell_area` containing the area of the grid cells.

        * If the dataset includes a CF-compliant grid mapping, and the LADiM file has
          lon/lat attributes, the particle coordinates will be transformed before mapping
          to the rasterized grid.

    :param weights: A tuple containing parameters to be summed up. If `None` is one of
        the parameters, a variable named `bincount` is created which contains the sum
        of particles within each cell.

    :return: An `xarray` dataset containing the rasterized data.
    """
    # Add edge info to grid dataset, if not present already
    grid_dset = add_edge_info(grid_dset)

    def get_edg(dset, vname):
        """Get bin edges of a 1D coordinate with contiguous cells"""
        bndname = dset[vname].attrs['bounds']
        return dset[bndname].values[:, 0].tolist() + [dset[bndname].values[-1, 1]]

    # Add areal info to grid dataset, if not present already
    grid_dset = add_area_info(grid_dset)

    # Convert projection of ladim dataset, if necessary
    particle_dset = change_ladim_crs(particle_dset, grid_dset)

    # Broadcast particle variables onto the particle_instance dimension
    for v in grid_dset.coords:
        if particle_dset[v].dims == ('particle', ):
            logger.info(f'Broadcasting variable {v} to particle_instance')
            particle_dset[v] = particle_dset[v].isel(particle=particle_dset['pid'])

    # Compute histogram data
    raster = from_particles(
        particles=particle_dset,
        bin_keys=[v for v in grid_dset.coords],
        bin_edges=[get_edg(grid_dset, v) for v in grid_dset.coords],
        vdims=weights,
    )

    # Merge histogram data and grid data
    logger.info("Copy attributes from grid dataset")
    new_raster = grid_dset.assign({v: raster.variables[v] for v in raster.data_vars})
    if 'time' in raster.coords:
        new_raster = new_raster.assign_coords(time=raster.coords['time'])
    _assign_georeference_to_data_vars(new_raster)

    # Copy attrs from ladim dataset
    logger.info("Copy attributes from particle dataset")
    for varname in set(new_raster.variables).intersection(particle_dset.variables):
        for k, v in particle_dset[varname].attrs.items():
            if k not in new_raster[varname].attrs:
                new_raster[varname].attrs[k] = v

    # CF conventions attribute
    new_raster.attrs['Conventions'] = "CF-1.8"

    return new_raster


def change_ladim_crs(ladim_dset, grid_dset):
    """Add crs coordinates to ladim dataset, taken from a grid dataset"""

    # Abort if there is no lat/lon coordinates in the ladim dataset
    if 'lat' not in ladim_dset.variables or 'lon' not in ladim_dset.variables:
        return ladim_dset

    from pyproj import Transformer

    crs_varname = _get_crs_varname(grid_dset)
    crs_xcoord = _get_crs_xcoord(grid_dset)
    crs_ycoord = _get_crs_ycoord(grid_dset)

    # Abort if there is no grid mapping
    if any(v is None for v in [crs_xcoord, crs_ycoord, crs_varname]):
        return ladim_dset

    target_crs = get_projection(grid_dset[crs_varname].attrs)
    transformer = Transformer.from_crs("epsg:4326", target_crs)
    x, y = transformer.transform(ladim_dset.lat.values, ladim_dset.lon.values)

    logger.info(f'Reproject particle coordinates from lat/lon to "{target_crs.to_proj4()}"')

    return ladim_dset.assign(**{
        crs_xcoord: xr.Variable(ladim_dset.lon.dims, x),
        crs_ycoord: xr.Variable(ladim_dset.lat.dims, y),
    })


def get_projection(grid_opts):
    from pyproj import CRS

    std_grid_opts = dict(
        false_easting=0,
        false_northing=0,
    )

    proj4str_dict = dict(
        latitude_longitude="+proj=latlon",
        polar_stereographic=(
            "+proj=stere +ellps=WGS84 "
            "+lat_0={latitude_of_projection_origin} "
            "+lat_ts={standard_parallel} "
            "+lon_0={straight_vertical_longitude_from_pole} "
            "+x_0={false_easting} "
            "+y_0={false_northing} "
        ),
        transverse_mercator=(
            "+proj=tmerc +ellps=WGS84 "
            "+lat_0={latitude_of_projection_origin} "
            "+lon_0={longitude_of_central_meridian} "
            "+k_0={scale_factor_at_central_meridian} "
            "+x_0={false_easting} "
            "+y_0={false_northing} "
        ),
        orthographic=(
            "+proj=ortho +ellps=WGS84 "
            "+lat_0={latitude_of_projection_origin} "
            "+lon_0={longitude_of_projection_origin} "
            "+x_0={false_easting} "
            "+y_0={false_northing} "
        ),
    )

    proj4str_template = proj4str_dict[grid_opts['grid_mapping_name']]
    proj4str = proj4str_template.format(**std_grid_opts, **grid_opts)
    return CRS.from_proj4(proj4str)

    pass


def add_area_info(dset):
    """Add cell area information to dataset coordinates, if not already present"""
    crs_varname = _get_crs_varname(dset)
    crs_xcoord = _get_crs_xcoord(dset)
    crs_ycoord = _get_crs_ycoord(dset)

    # Do nothing if crs info is unavailable
    if crs_varname is None:
        logger.info(f'Ignoring cell area, grid file lacks projection information')
        return dset

    # Do nothing if cell area exists
    stdnames = [dset[v].attrs.get('standard_name', '') for v in dset.variables]
    if "cell_area" in stdnames:
        cell_area_var = next(
            v for v in dset.variables
            if dset[v].attrs.get('standard_name', '') == "cell_area"
        )
        logger.info(f'Using cell area from variable {cell_area_var} in grid file')
        return dset

    x_bounds = dset[dset[crs_xcoord].attrs['bounds']].values
    y_bounds = dset[dset[crs_ycoord].attrs['bounds']].values
    x_diff = np.diff(x_bounds)[np.newaxis, :, 0]
    y_diff = np.diff(y_bounds)[:, np.newaxis, 0]
    grdmap = dset[crs_varname].attrs['grid_mapping_name']
    metric_projections = [
        'polar_stereographic', 'stereographic', 'orthographic', 'mercator',
        'transverse_mercator', 'oblique_mercator',
    ]

    logger.info(f'Computing cell area for grid mapping of type "{grdmap}"')

    # Compute cell area, or raise error if unknown mapping
    if grdmap == 'latitude_longitude':
        lon_diff_m, lat_diff_m = degree_diff_to_metric(
            lon_diff=x_diff, lat_diff=y_diff,
            reference_latitude=y_bounds.mean(axis=-1)[:, np.newaxis],
        )
        cell_area_data = lon_diff_m * lat_diff_m

    elif grdmap in metric_projections:
        cell_area_data = x_diff * y_diff
    else:
        raise NotImplementedError(f'Unknown grid mapping: {grdmap}')

    cell_area = xr.Variable(
        dims=(crs_ycoord, crs_xcoord),
        data=cell_area_data,
        attrs=dict(
            long_name="area of grid cell",
            standard_name="cell_area",
            units='m2',
        ),
    )

    return dset.assign(cell_area=cell_area)


def add_edge_info(dset):
    """Add edge information to dataset coordinates, if not already present"""
    dset_new = dset.copy()
    coords_with_bounds = [v for v in dset.coords if 'bounds' in dset[v].attrs]
    if coords_with_bounds:
        logger.info(f'Coordinates with bin edges in grid file: {coords_with_bounds}')

    coords_without_bounds = [v for v in dset.coords if 'bounds' not in dset[v].attrs]
    if coords_without_bounds:
        logger.info(f'Coordinates with bin centers in grid file: {coords_without_bounds}')

    for var_name in coords_without_bounds:
        bounds_name = var_name + '_bounds'
        edges_values = _edges(dset[var_name].values)
        bounds_var = xr.Variable(
            dims=dset[var_name].dims + ('bounds_dim', ),
            data=np.stack([edges_values[:-1], edges_values[1:]], axis=-1),
        )

        dset_new[var_name].attrs['bounds'] = bounds_name
        dset_new[bounds_name] = bounds_var

    return dset_new


def _edges(a):
    diff0 = a[1:] - a[:-1]
    extended = np.concatenate([a[:1] - diff0[:1], a, a[-1:] + diff0[-1:]])
    diff = extended[1:] - extended[:-1]
    if np.issubdtype(a.dtype, np.integer):
        half_diff = (0.5 * (diff + 1)).astype(a.dtype)
    elif np.issubdtype(a.dtype, np.datetime64):
        half_diff = (0.5 * (diff + np.array(1).astype(diff.dtype)))
    else:
        half_diff = 0.5 * diff
    edges = extended[:-1] + half_diff

    return edges


def _assign_georeference_to_data_vars(dset):
    crs_varname = _get_crs_varname(dset)
    crs_xcoord = _get_crs_xcoord(dset)
    crs_ycoord = _get_crs_ycoord(dset)

    for v in dset.data_vars:
        if crs_xcoord in dset[v].coords and crs_ycoord in dset[v].coords:
            dset[v].attrs['grid_mapping'] = crs_varname


def _get_crs_varname(dset):
    for v in dset.variables:
        if 'grid_mapping_name' in dset[v].attrs:
            return v
    return None


def _get_crs_xcoord(dset):
    for v in dset.coords:
        if 'standard_name' not in dset[v].attrs:
            continue
        s = dset[v].attrs['standard_name']
        if s == 'projection_x_coordinate' or s == 'grid_longitude' or s == 'longitude':
            return v
    return None


def _get_crs_ycoord(dset):
    for v in dset.coords:
        if 'standard_name' not in dset[v].attrs:
            continue
        s = dset[v].attrs['standard_name']
        if s == 'projection_y_coordinate' or s == 'grid_latitude' or s == 'latitude':
            return v
    return None


def from_particles(particles, bin_keys, bin_edges, vdims=(None,),
                   timevar_name='time', countvar_name='particle_count',
                   time_idx=None):

    # Handle variations on call signature
    if isinstance(particles, str):
        with xr.open_dataset(particles) as dset:
            return from_particles(dset, bin_keys, bin_edges, vdims)

    if countvar_name in particles:
        logger.info("Compute raster from sparse dataset")
        count = particles.variables[countvar_name].values
        indptr_name = particles.variables[bin_keys[0]].dims[0]
        indptr = np.cumsum(np.concatenate(([0], count)))
        slicefn = lambda tidx: particles.isel(
            {indptr_name: slice(indptr[tidx], indptr[tidx + 1])})
        tvals = particles[timevar_name].values
    elif timevar_name in particles.dims:
        logger.info("Compute raster from dense dataset")
        slicefn = lambda tidx: particles.isel({timevar_name: tidx})
        tvals = particles.variables[timevar_name].values
    else:
        logger.info("Compute raster from point cloud")
        slicefn = lambda tidx: particles
        tvals = None

    if time_idx is not None:
        slicefn_old = slicefn
        slicefn = lambda tidx: slicefn_old(time_idx)
        tvals = None

    return _from_particle(slicefn, tvals, bin_keys, bin_edges, vdims)


def _from_particle(slicefn, tvals, bin_keys, bin_edges, vdims):
    # Get the histogram for each time slot and property
    field_list = []
    for tidx in range(len(tvals) if tvals is not None else 1):
        logger.info(f"Load time index {tidx}")
        dset = slicefn(tidx)
        coords = [dset[k].values for k in bin_keys]
        weights = [None if w is None else dset[w].values for w in vdims]
        logger.info(f"Compute histogram for time index {tidx}")
        vals = [np.histogramdd(coords, bin_edges, weights=w)[0] for w in weights]
        field_list.append(vals)

    # Collect each histogram into an xarray variable
    logger.info("Merge histograms")
    field = np.array(field_list)
    xvars = {}
    for i, vdim in enumerate(vdims):
        vdim_name = 'bincount' if vdim is None else vdim
        kdims = ('time', ) + tuple(bin_keys)
        xvars[vdim_name] = xr.Variable(kdims, field[:, i])

    # Define bin center coordinates
    xcoords = {kdim: 0.5 * np.add(bin_edges[i][:-1], bin_edges[i][1:])
               for i, kdim in enumerate(bin_keys)}

    # Construct dataset
    if tvals is None:
        dset = xr.Dataset(xvars, xcoords).isel(time=0)
    else:
        xcoords['time'] = tvals
        dset = xr.Dataset(xvars, xcoords)

    return dset


def dt64_to_num(dates, units, calendar):
    """
    date2num(dates, units, calendar=None)

        Return numeric time values given a datetime64 array. The units
        of the numeric time values are described by the **units** argument
        and the **calendar** keyword. The datetime objects must
        be in UTC with no time-zone offset.

        The function encapsulates cftime.date2num, but allows numpy datetime64 inputs.

        **dates**: A numpy datetime64 array.

        **units**: a string of the form **<time units> since <reference time>**
        describing the time units. **<time units>** can be days, hours, minutes,
        seconds, milliseconds or microseconds. **<reference time>** is the time
        origin. **months_since** is allowed *only* for the **360_day** calendar.

        **calendar**: describes the calendar to be used in the time calculations.
        All the values currently defined in the
        [CF metadata convention](http://cfconventions.org)
        Valid calendars **'standard', 'gregorian', 'proleptic_gregorian'
        'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'**.
        Default is `None` which means the calendar associated with the rist
        input datetime instance will be used.

        returns a numpy array of integers
    """
    from cftime import date2num
    import datetime
    if dates.dtype.str.endswith('[ns]'):
        dates = dates.astype('datetime64[us]')
    py_dates = dates.tolist()
    py_datetime = [
        d if hasattr(d, 'hour')
        else datetime.datetime.combine(d, datetime.time())
        for d in py_dates
    ]
    return np.array(date2num(py_datetime, units, calendar))


def update_raster(dset_raster, chunk, bin_keys, weight_var=None):
    # Algorithm:
    # 1. Get bin_edges
    # 2. Do a quick survey of ladim_chunk to find max and min of each binning coordinate
    # 3. Construct a subset of bin_edges which includes the whole ladim chunk
    # 4. Construct a histogramdd
    # 5. Increment the appropriate slice in dset_raster with the histogramdd data

    def _get_chunk_vals(bin_key):
        values = chunk[bin_key]
        if not np.issubdtype(values.dtype, np.datetime64):
            return values
        return dt64_to_num(values, dset_raster[bin_key].units, dset_raster[bin_key].calendar)

    raster_varname = weight_var if weight_var else 'bincount'

    # --- Get bin edges from raster coordinate variable ---
    bin_edge_keys = (dset_raster[bin_key].bounds for bin_key in bin_keys)
    bin_edge_vars = (dset_raster[bin_edge_key] for bin_edge_key in bin_edge_keys)
    bin_edge_vals = [v[:, 0].tolist() + [v[-1, 1]] for v in bin_edge_vars]

    # --- Get coordinates from chunk, possibly converting dates to numeric values ---
    coords = [_get_chunk_vals(v) for v in bin_keys]
    weights = chunk[weight_var] if weight_var else None

    # --- Update raster using adaptive histogram --
    new_raster_val, idx = adaptive_histogram(coords, bin_edge_vals, weights=weights)
    previous_raster_val = dset_raster[raster_varname][idx]
    dset_raster[raster_varname][idx] = previous_raster_val + new_raster_val


def adaptive_histogram(sample, bins, **kwargs):
    idx = []
    bins_subset = []
    for coord, bin_edges in zip(sample, bins):
        digitized_min = np.digitize(np.min(coord), bin_edges)
        digitized_max = np.digitize(np.max(coord), bin_edges)
        idx_start = max(0, digitized_min - 1)
        idx_stop = min(len(bin_edges), digitized_max + 1)
        idx.append(slice(idx_start, idx_stop - 1))
        bins_subset.append(bin_edges[idx_start:idx_stop])

    rasterized_data = np.histogramdd(sample, bins_subset, **kwargs)[0]
    return rasterized_data, tuple(idx)


def _create_variable(dset, varname, datatype, dimensions, values):
    if np.issubdtype(datatype, np.datetime64):
        v = dset.createVariable(varname, np.float64, dimensions)
        v.units = f'milliseconds since 1970-01-01'
        v.calendar = 'standard'
        v[:] = (values - np.datetime64('1970-01-01T00:00:00.000')) / np.timedelta64(1, 'ms')
    else:
        dset.createVariable(varname, datatype, dimensions)[:] = values


def init_raster(dset_raster, bin_keys, bin_centers, bin_edges=None, weights=(), dset_ladim=None):
    bin_centers = [np.array(c) for c in bin_centers]

    if not bin_edges:
        bin_edges = [_edges(c) for c in bin_centers]

    dset_raster.createDimension('bounds_dim', 2)

    for k, c, e in zip(bin_keys, bin_centers, bin_edges):
        bnds_vals = np.stack([e[:-1], e[1:]], axis=-1)
        bnds_name = k + '_bounds'
        dset_raster.createDimension(k, len(c))
        _create_variable(dset_raster, k, datatype=c.dtype, dimensions=k, values=c)
        _create_variable(dset_raster, bnds_name, c.dtype, (k, 'bounds_dim'), bnds_vals)
        dset_raster[k].bounds = bnds_name

    dtypes = {k: np.float64 for k in weights}
    dtypes['bincount'] = np.int32

    if dset_ladim:
        vars_ladim = dset_ladim.variables
        dtypes_ladim = {v: vars_ladim[v].dtype for v in dtypes.keys() if v in vars_ladim}
        dtypes = {**dtypes, **dtypes_ladim}

    for n in weights + ('bincount', ):
        dset_raster.createVariable(n, dtypes[n], bin_keys)[:] = 0

    dset_raster.set_auto_maskandscale(False)


def ladim_chunks(ladim_datasets, varnames, max_rows=10000000):
    """An iterator for loading chunks of a multipart ladim dataset

    The function returns an iterator of chunks, where each chunk represents a time step
    of ladim data. Alternatively, each time step can be broken into even smaller chunks
    using the max_rows argument. The data is returned as a dict of numpy arrays.

    Variables can be either time-indexed, particle-indexed or particle_instance-indexed.
    In all cases, the output data is particle_instance-indexed.

    :param ladim_datasets: A sequence of xarray datasets
    :param varnames: Variable names to extract from the ladim datasets
    :param max_rows: Maximum number of rows per chunk
    :returns: A dict of numpy arrays, each of equal size
    """

    dim_inst = 'particle_instance'
    dim_time = 'time'
    dim_part = 'particle'
    var_pidx = 'pid'
    var_pcnt = 'particle_count'

    for dset in ladim_datasets:
        varnames_time = [v for v in varnames if dset[v].dims == (dim_time,)]
        varnames_part = [v for v in varnames if dset[v].dims == (dim_part,)]
        varnames_inst = [v for v in varnames if dset[v].dims == (dim_inst,)]

        idx_inst_prev = 0

        for idx_time in range(dset.dims[dim_time]):
            cum_pinst = idx_inst_prev + dset[var_pcnt][idx_time].values.item()
            while idx_inst_prev < cum_pinst:
                idx_inst_next = min(idx_inst_prev + max_rows, cum_pinst)
                idx_inst = range(idx_inst_prev, idx_inst_next)
                idx_part = dset[var_pidx][idx_inst].values
                data_part = {v: dset[v].isel({dim_part: idx_part}).values for v in varnames_part}
                data_inst = {v: dset[v].isel({dim_inst: idx_inst}).values for v in varnames_inst}
                data_time = {
                    v: np.broadcast_to(
                        dset[v].isel({dim_time: idx_time}).values,
                        (idx_inst_next - idx_inst_prev),
                    )
                    for v in varnames_time
                }
                yield {**data_time, **data_part, **data_inst}
                idx_inst_prev = idx_inst_next


def parse_args(args):
    import argparse
    parser = argparse.ArgumentParser(
        prog='ladim_raster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert LADiM output data to netCDF raster format.",
        epilog=(
            "The script requires a pre-defined netCDF raster file with a\n"
            "variable named `bincount`, initialized to zero. The script\n"
            "populates this variable with particle count data from the\n"
            "LADiM file.\n\n"
            "The coordinates of `bincount` define the center of each\n"
            "bin. The bin edges are halfway between the bin centers,\n"
            "unless a CF-compliant `bounds` attribute is supplied. The\n"
            "name of each coordinate must match a variable name in the\n"
            "LADiM file.\n\n"
            "If a weighted sum is desired instead of a bin count, the\n"
            "`bincount` variable should be renamed to match the name of\n"
            "the weighting variable. For instance, if a weighted sum of\n"
            "the LADiM variable `super` is desired, the `bincount`\n"
            "variable in the raster dataset should be renamed `super`\n"
            "before applying the script.\n\n"
            "Sample raster file:\n\n"
            "dimensions:\n"
            "  X = 500;\n"
            "  Y = 600;\n"
            "  time = 10;\n"
            "variables:\n"
            '  int bincount(time, X, Y) ;  // Initialized to zero\n'
            '  float X(X) ;      // bin centers for X coordinate\n'
            '  float Y(Y) ;      // bin centers for Y coordinate\n'
            "  int time(time) ;  // one entry per time coordinate\n"
            '    time:units = "seconds since 1970-01-01" ;\n'
        ),
    )

    parser.add_argument(
        "raster_file",
        help="netCDF raster file")

    parser.add_argument(
        "ladim_file",
        nargs='+',
        help="output file(s) from LADiM, glob patterns allowed (*?[])",
    )

    parsed_args = parser.parse_args(args)

    # Expand glob patterns
    parsed_args.ladim_file = _glob(parsed_args.ladim_file)
    if len(parsed_args.ladim_file) == 0:
        parser.print_usage()
        print(f'error: No ladim_file found')
        raise SystemExit(2)

    return parsed_args


def _glob(fnames):
    g = []
    for fname in fnames:
        if set(fname).intersection(set('*?[]')):
            g += sorted(glob.glob(fname))
        else:
            g.append(fname)
    return g


def _xr_iter(fnames):
    for fname in fnames:
        with xr.open_dataset(fname) as dset:
            yield dset


def rasterize(raster, particles):
    bin_keys = raster['bincount'].dimensions
    weights = ()

    for chunk in ladim_chunks(particles, list(bin_keys) + list(weights)):
        for weight_var in weights + (None, ):
            update_raster(raster, chunk, bin_keys, weight_var)


def main():
    import sys
    args = parse_args(sys.argv[1:])

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    import netCDF4 as nc
    with nc.Dataset(args.raster_file, 'a') as raster_dset:
        rasterize(raster_dset, particles=_xr_iter(args.ladim_file))


if __name__ == '__main__':
    main()
