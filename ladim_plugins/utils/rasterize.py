import xarray as xr
import netCDF4 as nc
import numpy as np
from ladim_plugins.release.makrel import degree_diff_to_metric
import logging
import glob
import typing


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
        edges_values = get_edges(dset[var_name].values)
        bounds_var = xr.Variable(
            dims=dset[var_name].dims + ('bounds_dim', ),
            data=np.stack([edges_values[:-1], edges_values[1:]], axis=-1),
        )

        dset_new[var_name].attrs['bounds'] = bounds_name
        dset_new[bounds_name] = bounds_var

    return dset_new


def get_edges(a):
    half_offset = 0.5 * (a[1:] - a[:-1])
    first_edge = a[0] - half_offset[0]
    last_edge = a[-1] + half_offset[-1]
    mid_edges = a[:-1] + half_offset
    return np.concatenate([[first_edge], mid_edges, [last_edge]])


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


def ladim_iterator(ladim_dsets):
    for dset in ladim_dsets:
        pcount_cum = np.concatenate([[0], np.cumsum(dset.particle_count.values)])

        for tidx in range(dset.dims['time']):
            logger.info(f'Read time step {dset.time[tidx].values}')
            iidx = slice(pcount_cum[tidx], pcount_cum[tidx + 1])
            logger.info(f'Number of particles: {iidx.stop - iidx.start}')
            pidx = xr.Variable('particle_instance', dset.pid[iidx].values)
            ttidx = xr.Variable('particle_instance', np.broadcast_to(tidx, (len(pidx), )))
            ddset = dset.isel(
                time=ttidx,
                particle_instance=iidx,
                particle=pidx,
            )
            ddset = ddset.assign(instance_offset=dset.instance_offset + iidx.start)
            yield ddset


def _ladim_iterator_read_variable(dset, varname, tidx, iidx, pidx):
    v = dset.variables[varname]
    first_dim = v.dims[0]
    if first_dim == 'particle_instance':
        return v[iidx].values
    elif first_dim == 'particle':
        return v[pidx].values
    elif first_dim == 'time':
        return v[tidx].values
    else:
        raise ValueError(f'Unknown dimension type: {first_dim}')


class LadimInputStream:
    def __init__(self, spec):
        # Convert input spec to a sequence
        if isinstance(spec, tuple) or isinstance(spec, list):
            specs = spec
        else:
            specs = [spec]

        # Expand glob patterns in spec
        self.datasets = []
        for s in specs:
            if isinstance(s, str):
                self.datasets += sorted(glob.glob(s))
            else:
                self.datasets.append(s)
        logger.info(f'Number of input datasets: {len(self.datasets)}')

        self._filter = lambda chunk: chunk
        self._weights = None

        self._dataset_iterator = None
        self._dataset_current = xr.Dataset()
        self._dataset_mustclose = False
        self.ladim_iter = None
        self._reset_ladim_iterator()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dataset_mustclose:
            self._dataset_current.close()

    def close(self):
        self._dataset_current.close()

    def seek(self, pos):
        if pos != 0:
            raise NotImplementedError
        self._reset_ladim_iterator()

    def _reset_ladim_iterator(self):
        if self._dataset_mustclose:
            self._dataset_current.close()

        def dataset_iterator():
            for spec in self.datasets:
                if isinstance(spec, str):
                    logger.info(f'Open input dataset {spec}')
                    with xr.open_dataset(spec) as dset:
                        logger.info(f'Number of particle instances: {dset.dims["particle_instance"]}')
                        self._dataset_current = dset
                        self._dataset_mustclose = True
                        yield dset
                else:
                    logger.info(f'Enter new dataset')
                    self._dataset_current = spec
                    self._dataset_mustclose = False
                    logger.info(f'Number of particle instances: {spec.dims["particle_instance"]}')
                    yield spec

        self._dataset_iterator = dataset_iterator()
        self.ladim_iter = ladim_iterator(self._dataset_iterator)

    @property
    def filter(self):
        return self._filter

    @property
    def weights(self):
        return self._weights

    @filter.setter
    def filter(self, spec):
        if spec is None:
            return
        elif isinstance(spec, str):
            self._filter = get_filter_func(spec)
        elif callable(spec):
            self._filter = spec
        else:
            raise TypeError(f'Unknown type: {type(spec)}')

    @weights.setter
    def weights(self, spec):
        if spec is None:
            return
        elif isinstance(spec, str):
            self._weights = get_weight_func(spec)
        elif callable(spec):
            self._weights = spec
        else:
            raise TypeError(f'Unknown type: {type(spec)}')

    def find_limits(self, resolution):
        def iterate_datasets() -> typing.Iterable:
            for spec in self.datasets:
                if isinstance(spec, str):
                    logger.info(f'Open input dataset {spec}')
                    with xr.open_dataset(spec) as ddset:
                        yield ddset
                else:
                    logger.info(f'Enter new dataset')
                    yield spec

        def t64conv(timedelta_or_other):
            try:
                t64val, t64unit = timedelta_or_other
                return np.timedelta64(t64val, t64unit)
            except TypeError:
                return timedelta_or_other

        # Align to wholenumber resolution
        def align(val_raw, res_raw):
            if np.issubdtype(np.array(res).dtype, np.timedelta64):
                val_posix = (val_raw - np.datetime64('1970-01-01')).astype('timedelta64[us]')
                res_posix = res.astype('timedelta64[us]')
                ret_posix = (val_posix.astype('i8') // res_posix.astype('i8')) * res_posix
                return np.datetime64('1970-01-01') + ret_posix
            else:
                return np.array((val_raw // res_raw) * res_raw).item()

        varnames = resolution.keys()
        logger.info("Limits are not given, compute automatically from input file")
        minvals = {k: [] for k in varnames}
        maxvals = {k: [] for k in varnames}
        for dset in iterate_datasets():
            for k in varnames:
                res = t64conv(resolution[k])
                minval = align(dset.variables[k].min().values, res)
                maxval = align(dset.variables[k].max().values + res, res)
                logger.info(f'Limits for {k} in current dataset: [{minval}, {maxval}]')
                minvals[k].append(minval)
                maxvals[k].append(maxval)

        lims = {k: [np.min(minvals[k]), np.max(maxvals[k])] for k in varnames}
        for k in varnames:
            logger.info(f'Final limits for {k}: [{lims[k][0]}, {lims[k][1]}]')
        return lims

    def read(self):
        try:
            chunk = next(self.ladim_iter)
            chunk = self.filter(chunk)
            if self.weights:
                chunk = chunk.assign(weights=self.weights(chunk))
            return chunk
        except StopIteration:
            return None

    def chunks(self) -> typing.Iterable:
        chunk = self.read()
        while chunk is not None:
            yield chunk
            chunk = self.read()


def get_filter_func(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def filter_fn(chunk):
        args = [chunk[n].values for n in ex.input_names]
        idx = ex.run(*args)
        return chunk.isel(particle_instance=idx)
    return filter_fn


def get_weight_func(spec):
    import numexpr
    ex = numexpr.NumExpr(spec)

    def weight_fn(chunk):
        args = [chunk[n].values for n in ex.input_names]
        return xr.Variable('particle_instance', ex.run(*args))

    return weight_fn


class RasterOutputStream:
    def __init__(self, spec):
        if isinstance(spec, str):
            logger.info(f'Open output dataset {spec}')
            self.dataset = nc.Dataset(spec, 'w')
            self._must_close = True
        else:
            self.dataset = spec
            self._must_close = False
        self.histogram_varname = 'histogram'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._must_close:
            self.close()

    def close(self):
        self.dataset.close()

    def add_histogram(self, indices, values):
        v = self.dataset.variables[self.histogram_varname]
        v[indices] += values.astype(v.dtype)

    def write_coord(self, name, data, attrs=None):
        data = np.array(data)
        logger.info(f'Write coordinate {name}({len(data)}) to output file')

        self.dataset.createDimension(name, len(data))

        if np.issubdtype(data.dtype, np.datetime64):
            offset = (data - np.datetime64('1970-01-01', 'us')).astype('i8')
            v = self.dataset.createVariable(name, 'i8', name)
            v[:] = offset
            v.setncatts(dict(
                units="microseconds since 1970-01-01",
                calendar="proleptic_gregorian",
            ))
        else:
            self.dataset.createVariable(name, data.dtype, name)[:] = data
        self.dataset.variables[name].set_auto_maskandscale(False)
        if attrs:
            self.dataset.variables[name].setncatts(attrs)

    def write_coords(self, hist):
        for name, coord_data in hist.coords.items():
            self.write_coord(name, coord_data['centers'])

    def write_vars(self, hist):
        logger.info(f'Initialize output variable "{self.histogram_varname}"')
        dims = tuple(hist.coords.keys())
        v = self.dataset.createVariable(self.histogram_varname, 'f4', dims)
        v.set_auto_maskandscale(False)
        v[:] = 0


class Histogrammer:
    def __init__(self, resolution, limits):
        self.resolution = resolution
        self.limits = limits
        self.weights = dict(bincount=None)
        self.coords = self._get_coords()

    def _get_coords(self):
        crd = dict()
        for k, v in self.resolution.items():
            start, stop = self.limits[k]

            # Check if limits is a datestring
            if isinstance(start, str) and isinstance(stop, str):
                try:
                    start, stop = np.array([start, stop]).astype('datetime64')
                except ValueError:
                    pass

            # Check if resolution is a timedelta specified as [value, unit]
            if np.issubdtype(np.array(start).dtype, np.datetime64):
                try:
                    t64val, t64unit = v
                    v = np.timedelta64(t64val, t64unit)
                except TypeError:
                    pass

            centers = np.arange(start, stop + v, v)
            if centers[-1] > stop:
                centers = centers[:-1]
            edges = get_edges(centers)
            crd[k] = dict(centers=centers, edges=edges)
        return crd

    def make(self, chunk):
        coord_names = list(self.coords.keys())
        bins = [self.coords[k]['edges'] for k in coord_names]
        indices = tuple(slice(None) for _ in range(len(coord_names)))
        coords = [chunk[k].values for k in coord_names]

        if 'weights' in chunk.variables:
            weights = chunk.weights.values
        else:
            weights = None

        values, _ = np.histogramdd(coords, bins, weights=weights)
        yield dict(indices=indices, values=values)


def ladim_conc(
        resolution, input_file, output_file, limits=None, afilter=None,
        weights=None):
    # 1. Opprette output-fil
    # 2. Åpne input-fil(er) som en chunk-strøm
    # 3. Pipeline: Input chunk-strøm til funksjon, som gir output chunk-strøm
    # 4. Lagre chunk-strøm til output-fil

    with LadimInputStream(input_file) as dset_in:
        dset_in.filter = afilter
        dset_in.weights = weights

        if limits is None:
            limits = dict()
        resolution_remaining = {k: v for k, v in resolution.items() if k not in limits}
        if resolution_remaining:
            limits = {**limits, **dset_in.find_limits(resolution_remaining)}

        hist = Histogrammer(
            resolution=resolution,
            limits=limits,
        )

        with RasterOutputStream(output_file) as dset_out:
            dset_out.write_coords(hist)
            dset_out.write_vars(hist)
            for chunk_in in dset_in.chunks():
                for chunk_out in hist.make(chunk_in):
                    dset_out.add_histogram(**chunk_out)


def main2():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert LADiM output data to netCDF raster format.',
    )
    parser.add_argument("config_file", help="configuration file")

    args = parser.parse_args()

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Load config file
    logger.info(f'Load config file {args.config_file}')
    import yaml
    with open(args.config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for line in yaml.safe_dump(config).split('\n'):
        logger.info(line)

    # Invoke main function
    ladim_conc(
        input_file=config['input_file'],
        output_file=config['output_file'],
        resolution=config['resolution'],
        limits=config.get('limits', None),
        afilter=config.get('filter', None),
        weights=config.get('weights', None),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert LADiM output data to netCDF raster format.',
    )

    parser.add_argument("ladim_file", help="output file from LADiM")
    parser.add_argument(
        "grid_file",
        help="netCDF file containing the bins. Any coordinate variable in the file "
             "which match the name of a LADiM variable is used.")

    parser.add_argument(
        "raster_file",
        help="output file name",
    )

    parser.add_argument(
        "--weights",
        nargs='+',
        metavar='varname',
        help="weighting variables",
        default=(),
    )

    args = parser.parse_args()
    weights = (None, ) + tuple(args.weights)

    # Initialize logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.info(f'Open grid file {args.grid_file}')
    grid_dset = xr.load_dataset(args.grid_file)

    # --- Use either single-file or multi-file approach ---

    import glob
    import os
    ladim_files = sorted(glob.glob(args.ladim_file))

    if len(ladim_files) == 0:
        raise IOError(f'File "{ladim_files}" not found')

    elif len(ladim_files) == 1:
        logger.info(f'Open particle file {ladim_files[0]}')
        with xr.open_dataset(ladim_files[0]) as ladim_dset:
            raster = ladim_raster(ladim_dset, grid_dset, weights=weights)
            logger.info(f'Save raster to {args.raster_file}')
            raster.to_netcdf(args.raster_file)

    else:
        rfile_base, rfile_ext = os.path.splitext(args.raster_file)
        rfiles = [f'{rfile_base}_{i:04}{rfile_ext}' for i in range(len(ladim_files))]
        for ladim_file, raster_file in zip(ladim_files, rfiles):
            logger.info(f'Open particle file {ladim_file}')
            with xr.open_dataset(ladim_file) as ladim_dset:
                raster = ladim_raster(ladim_dset, grid_dset, weights=weights)
                logger.info(f'Save raster to {raster_file}')
                raster.to_netcdf(raster_file)


if __name__ == '__main__':
    main()
