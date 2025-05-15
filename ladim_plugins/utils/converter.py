import numpy as np


def ladim_file_to_sqlite(fname_in_pattern, fname_out):
    import glob
    import xarray as xr
    import sqlite3
    import logging
    logger = logging.getLogger(__name__)

    fnames_in = sorted(glob.glob(fname_in_pattern))

    logger.info(f'Create file {fname_out}')
    with sqlite3.connect(fname_out) as con:
        cur = con.cursor()

        with xr.open_dataset(fnames_in[0], decode_times=False) as dset:
            logger.info('Create tables')
            add_particle_table(dset, cur)
            add_instance_table(dset, cur)
            add_particle_values(dset, cur)

        for ladim_fname in fnames_in:
            logger.info(f'Add particle data from {ladim_fname}')
            with xr.open_dataset(ladim_fname, decode_times=False) as dset:
                add_instance_values(dset, cur)


def to_sqlite(dset, con):
    cur = con.cursor()
    add_particle_table(dset, cur)
    add_instance_table(dset, cur)
    add_particle_values(dset, cur)
    add_instance_values(dset, cur)


def add_particle_table(dset, cur):
    particle_cols = [k for k, v in dset.variables.items() if v.dims == ('particle', )]
    cmd = (
        "CREATE TABLE IF NOT EXISTS particle ("
        + ",".join([f'{p} REAL NOT NULL' for p in particle_cols])
        + ");"
    )
    cur.execute(cmd)


def add_instance_table(dset, cur):
    instance_cols = [k for k, v in dset.variables.items() if v.dims == ('particle_instance', )]
    cmd = (
        "CREATE TABLE IF NOT EXISTS particle_instance ("
        + "time REAL NOT NULL,"
        + ",".join([f'{c} REAL NOT NULL' for c in instance_cols])
        + ");"
    )
    cur.execute(cmd)


def add_particle_values(dset, cur):
    particle_cols = [k for k, v in dset.variables.items() if v.dims == ('particle', )]
    values = np.array([dset[c].values for c in particle_cols])
    cmd = "INSERT INTO particle VALUES(" + ",".join(["?"] * len(particle_cols)) + ")"
    cur.executemany(cmd, values.T.tolist())


def add_instance_values(dset, cur):
    cum_count = np.concatenate([[0], np.cumsum(dset.particle_count.values)])
    cols = [k for k, v in dset.variables.items() if v.dims == ('particle_instance', )]
    cmd = "INSERT INTO particle_instance VALUES(" + ",".join(["?"] * (1 + len(cols))) + ")"

    for tidx in range(len(cum_count) - 1):
        iidx = slice(cum_count[tidx], cum_count[tidx + 1])
        tvals = np.repeat(dset['time'][tidx].values, iidx.stop - iidx.start)
        values = np.array([tvals] + [dset[c][iidx].values for c in cols])
        cur.executemany(cmd, values.T.tolist())


def ladim_file_to_parquet(fname_in_pattern, fname_out):
    from pathlib import Path
    import glob
    import logging
    logger = logging.getLogger(__name__)

    paths_in = sorted(Path(p) for p in glob.iglob(fname_in_pattern))
    path_out = Path(fname_out)
    if len(paths_in) == 0:
        raise OSError(f'No input files found: {fname_in_pattern}')
    if not path_out.is_dir():
        raise OSError(f'Output path does not exist: {path_out}')
    if any(path_out.iterdir()):
        raise OSError(f'Output directory is not empty: {path_out}')

    (path_out / 'particle').mkdir(exist_ok=False, parents=False)
    outfile_pattern = str(path_out / 'particle' / 'part.{:06}.parquet')
    for i, chunk in enumerate(load_ladim_particle_chunks(paths_in[0])):
        fname_chunk = outfile_pattern.format(i)
        logger.info(f'Write chunk to file: {fname_chunk}')
        chunk.to_parquet(path=fname_chunk, engine='pyarrow', compression='snappy', index=True)

    (path_out / 'instance').mkdir(exist_ok=False, parents=False)
    outfile_pattern = str(path_out / 'instance' / 'part.{:06}.parquet')
    for i, chunk in enumerate(load_ladim_instance_chunks(paths_in)):
        fname_chunk = outfile_pattern.format(i)
        logger.info(f'Write chunk to file: {fname_chunk}')
        chunk.to_parquet(path=fname_chunk, engine='pyarrow', compression='snappy', index=False)


def load_ladim_instance_chunks(paths):
    import xarray as xr

    for path in paths:
        with xr.open_dataset(path, decode_cf=False, engine='h5netcdf') as dset:
            iterator = load_ladim_instance_chunks_from_dataset(dset)
            for chunk in iterator:
                yield chunk


def load_ladim_instance_chunks_from_dataset(dset):
    ddset = dset.drop_dims(['time', 'particle']).drop_vars('instance_offset')
    start = 0
    for tidx in range(dset.sizes['time']):
        stop = start + dset['particle_count'][tidx].item()
        subset = ddset.isel(particle_instance=slice(start, stop))
        df = subset.to_dataframe()
        df['time'] = dset['time'][tidx].item()
        yield df
        start = stop


def load_ladim_particle_chunks(path):
    import xarray as xr
    with xr.open_dataset(path, decode_cf=False, engine='h5netcdf') as dset:
        iterator = load_ladim_particle_chunks_from_dataset(dset)
        for chunk in iterator:
            yield chunk


def load_ladim_particle_chunks_from_dataset(dset):
    ddset = dset.drop_dims(['time', 'particle_instance']).drop_vars('instance_offset')
    sz = int(1e7)
    start = 0
    num_particles = dset.sizes['particle']
    while start < num_particles:
        stop = start + sz
        subset = ddset.isel(particle=slice(start, stop))
        df = subset.to_dataframe()
        yield df
        start = stop