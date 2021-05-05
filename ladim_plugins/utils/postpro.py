import xarray as xr
import numpy as np
from ladim_plugins.release.makrel import degree_diff_to_metric


def ladim_raster(input_dset, grid_dset, weights=(None,)):
    # Add edge info to grid dataset, if not present already
    grid_dset = add_edge_info(grid_dset)

    def get_edg(dset, varname):
        """Get bin edges of a 1D coordinate with contiguous cells"""
        bndname = dset[varname].attrs['bounds']
        return dset[bndname].values[:, 0].tolist() + [dset[bndname].values[-1, 1]]

    # Add areal info to grid dataset, if not present already
    grid_dset = add_area_info(grid_dset)

    # Compute histogram data
    raster = from_particles(
        particles=input_dset,
        bin_keys=[v for v in grid_dset.coords],
        bin_edges=[get_edg(grid_dset, v) for v in grid_dset.coords],
        vdims=weights,
    )

    # Merge histogram data and grid data
    new_raster = grid_dset.assign({v: raster.variables[v] for v in raster.data_vars})
    if 'time' in raster.coords:
        new_raster = new_raster.assign_coords(time=raster.coords['time'])
    _assign_georeference_to_data_vars(new_raster)

    # Copy attrs from ladim dataset
    for varname in set(new_raster.variables).intersection(input_dset.variables):
        for k, v in input_dset[varname].attrs.items():
            if k not in new_raster[varname].attrs:
                new_raster[varname].attrs[k] = v

    # CF conventions attribute
    new_raster.attrs['Conventions'] = "CF-1.8"

    return new_raster


def add_area_info(dset):
    """Add cell area information to dataset coordinates, if not already present"""
    crs_varname = _get_crs_varname(dset)
    crs_xcoord = _get_crs_xcoord(dset)
    crs_ycoord = _get_crs_ycoord(dset)

    # Do nothing if crs info is unavailable
    if crs_varname is None:
        return dset

    # Do nothing if cell area exists
    stdnames = [dset[v].attrs.get('standard_name', '') for v in dset.variables]
    if "cell_area" in stdnames:
        return dset

    # Raise error if unknown crs mapping
    if dset[crs_varname].attrs['grid_mapping_name'] != 'latitude_longitude':
        raise NotImplementedError

    lon_bounds = dset[dset[crs_xcoord].attrs['bounds']].values
    lat_bounds = dset[dset[crs_ycoord].attrs['bounds']].values

    lon_diff_m, lat_diff_m = degree_diff_to_metric(
        lon_diff=np.diff(lon_bounds)[np.newaxis, :, 0],
        lat_diff=np.diff(lat_bounds)[:, np.newaxis, 0],
        reference_latitude=lat_bounds.mean(axis=-1)[:, np.newaxis],
    )

    cell_area = xr.Variable(
        dims=(crs_ycoord, crs_xcoord),
        data=lon_diff_m * lat_diff_m,
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
    coords_without_bounds = [v for v in dset.coords if 'bounds' not in dset[v].attrs]
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
    mid = 0.5 * (a[:-1] + a[1:])
    return np.concatenate([mid[:1] - (a[1] - a[0]), mid, mid[-1:] + a[-1] - a[-2]])


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
        count = particles.variables[countvar_name].values
        indptr_name = particles.variables[bin_keys[0]].dims[0]
        indptr = np.cumsum(np.concatenate(([0], count)))
        slicefn = lambda tidx: particles.isel(
            {indptr_name: slice(indptr[tidx], indptr[tidx + 1])})
        tvals = particles[timevar_name].values
    elif timevar_name in particles.dims:
        slicefn = lambda tidx: particles.isel({timevar_name: tidx})
        tvals = particles.variables[timevar_name].values
    else:
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
        dset = slicefn(tidx)
        coords = [dset[k].values for k in bin_keys]
        weights = [None if w is None else dset[w].values for w in vdims]
        vals = [np.histogramdd(coords, bin_edges, weights=w)[0] for w in weights]
        field_list.append(vals)

    # Collect each histogram into an xarray variable
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert output data from LADiM to raster format.',
    )

    parser.add_argument("ladim_file", help="output file from LADiM")
    parser.add_argument(
        "grid_file",
        help="NetCDF file containing bin edges. Any coordinate variable in the file "
             "which match the name of a LADiM variable is used.")
    parser.add_argument("raster_file", help="output raster file name, netCDF format")
    parser.add_argument("--weights", nargs='+', metavar='varname', help="weighting variables")

    args = parser.parse_args()
    if args.weights is None:
        weights = (None, )
    else:
        weights = (None, ) + tuple(args.weights)

    with xr.open_dataset(args.ladim_file) as ladim_dset:
        with xr.open_dataset(args.grid_file) as grid_dset:
            raster = ladim_raster(ladim_dset, grid_dset, weights=weights)

    raster.to_netcdf(args.raster_file)
