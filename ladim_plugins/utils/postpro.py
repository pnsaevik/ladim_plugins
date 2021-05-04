import xarray as xr
import numpy as np


def ladim_raster(input_dset, grid_dset, weights=(None,)):
    raster = from_particles(
        particles=input_dset,
        bin_keys=[v for v in grid_dset.coords],
        bin_edges=[_edges(grid_dset[v].values) for v in grid_dset.coords],
        vdims=weights,
    )

    new_raster = grid_dset.assign(raster.data_vars)
    return new_raster


def _edges(a):
    mid = 0.5 * (a[:-1] + a[1:])
    return np.concatenate([mid[:1] - (a[1] - a[0]), mid, mid[-1:] + a[-1] - a[-2]])


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

    with xr.open_dataset(args.ladim_file) as ladim_dset:
        with xr.open_dataset(args.grid_file) as grid_dset:
            raster = ladim_raster(ladim_dset, grid_dset)

    raster.to_netcdf(args.raster_file)
