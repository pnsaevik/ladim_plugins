import numpy as np


def ladim_file_to_sqlite(fname_in_pattern, fname_out):
    import glob
    import xarray as xr
    import sqlite3
    import logging
    logger = logging.getLogger(__name__)

    fnames_in = sorted(glob.glob(fname_in_pattern))

    logger.info(f'Create file {fname_out}')

    con = sqlite3.connect(fname_out)
    try:
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
        
        cur.close()

    finally:
        con.close()

def to_sqlite(dset, con):
    cur = con.cursor()
    add_particle_table(dset, cur)
    add_instance_table(dset, cur)
    add_particle_values(dset, cur)
    add_instance_values(dset, cur)
    cur.close()


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
