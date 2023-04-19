import numpy as np


def to_sqlite(dset, con):
    cur = con.cursor()

    particle_cols = [k for k, v in dset.variables.items() if v.dims == ('particle', )]
    cmd = (
        "CREATE TABLE IF NOT EXISTS particle ("
        + ",".join([f'{p} REAL NOT NULL' for p in particle_cols])
        + ");"
    )
    cur.execute(cmd)

    instance_cols = [k for k, v in dset.variables.items() if v.dims == ('particle_instance', )]
    cmd = (
        "CREATE TABLE IF NOT EXISTS particle_instance ("
        + "time REAL NOT NULL,"
        + ",".join([f'{c} REAL NOT NULL' for c in instance_cols])
        + ");"
    )
    cur.execute(cmd)

    # cum_count = np.concatenate([[0], np.cumsum(dset.particle_count.values)])
    values = np.array([dset[c].values for c in particle_cols])
    cmd = "INSERT INTO particle VALUES(" + ",".join(["?"] * len(particle_cols)) + ")"
    cur.executemany(cmd, values.T.tolist())
